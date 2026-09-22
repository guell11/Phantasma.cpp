# ID_003 dense tree mask reference

`common_tree_draft_dense_mask_build()` is the CPU correctness oracle for the ID_003 additive tree visibility mask. It consumes the accepted ID_001 packed topology and the accepted ID_002 ancestor closure without introducing another topology representation.

For `N = topology.n_nodes`, output storage is a contiguous row-major `float` matrix with shape `[N, N]`. Query node `i` selects row `i`; key node `j` selects column `j`; the flat offset is `i * N + j`. The caller supplies `value_count`, which must be at least `N * N`. Values beyond the first `N * N` elements are not touched.

The exact reference value is:

```text
D[i,j] = 0.0f  when tree_id[i] == tree_id[j] and ancestor[i,j] is set
         -inf   otherwise
```

The ID_002 ancestor bitset is inclusive, so every node sees itself. A node sees all and only its ancestors in the same tree. Siblings cannot see one another, descendants cannot be seen by ancestors, and nodes in different trees are masked even if a corrupted ancestor bit happens to be set across trees.

The builder first validates ID_001 topology. For non-empty input it then requires an ID_002 bitset with at least `N * ceil(N / 32)` words and output storage with at least `N * N` floats. Size products are checked for `size_t` overflow before pointer or capacity checks that depend on them. Empty forests succeed with null ancestor/output pointers because they require no storage. Buffer-capacity failures occur before any mask element is written.

Example for `parent = [-1, 0, 0, 1, -1, 4]` and `tree_id = [7, 7, 7, 7, 42, 42]`:

```text
query\\key   0     1     2     3     4     5
0            0    -inf  -inf  -inf  -inf  -inf
1            0     0    -inf  -inf  -inf  -inf
2            0    -inf   0    -inf  -inf  -inf
3            0     0    -inf   0    -inf  -inf
4           -inf  -inf  -inf  -inf   0    -inf
5           -inf  -inf  -inf  -inf   0     0
```

This dense matrix is intentionally a CPU/reference surface. Production compact masks or accelerator predicates can compare against it, but ID_003 does not select or implement those successor paths.
