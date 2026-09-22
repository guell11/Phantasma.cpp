# ID_002 ancestor closure construction

ID_002 consumes the `common_tree_draft_topology` ABI from ID_001 and emits a row-major packed ancestor matrix using `uint32_t` words.

For `N` nodes, `words_per_row = ceil(N / 32)`. Row `i` starts at `words + i * words_per_row`; logical column `j` is stored in word `j / 32`, bit `j % 32`, with the least-significant bit representing the lowest node index in each word. A set bit means `j` is node `i` itself or an ancestor of `i`. Bits at columns `j >= N` in the final word remain zero.

Because ID_001 requires `parent[i] < i`, row `i` can be built by copying the already-complete row of `parent[i]` and then setting bit `i`. Roots start from an all-zero row and set only their own bit. This costs `O(N * ceil(N / 32))` packed-word operations and does not follow a parent chain once per output bit.

The CPU/reference entry point is `common_tree_draft_ancestor_build()`. The caller owns the output storage and supplies its word capacity. Empty forests require no storage. Non-empty topologies are validated through ID_001 before construction, and undersized or null output buffers fail without writing a partial closure.

The same recurrence is GPU-friendly. Nodes at one depth may be processed in parallel after all rows at the previous depth are visible. A CUDA/Triton implementation may therefore launch one depth wave at a time, or use an equivalent barrier that establishes parent-row completion before child-row reads. No atomic updates are required because each row has a single writer. Independent packed forests may be built in separate launches or streams as long as their output ranges do not overlap.

Example for `parent = [-1, 0, 0, 1]`:

```text
row 0: 0001
row 1: 0011
row 2: 0101
row 3: 1011
```

The bit strings above are shown with node 0 at the right for readability; in memory node 0 is bit 0 of the first `uint32_t` word.
