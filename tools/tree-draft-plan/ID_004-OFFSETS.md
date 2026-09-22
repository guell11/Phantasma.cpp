# ID_004 packed forest offsets

`common_tree_draft_forest_offsets_build()` consumes the accepted ID_001 `common_tree_draft_topology` ABI and an explicit `int32` node count `N_b` for each ragged batch entry.

For `B` entries, the output is a contiguous `int32` array with shape `[B + 1]`:

```text
offset[0] = 0
offset[b + 1] = offset[b] + N_b
```

Every `N_b` is non-negative. Empty entries are valid and produce repeated offsets. Accumulation is checked before each addition and cannot exceed `INT32_MAX`; the final offset must equal `topology.n_nodes`. A zero-entry forest therefore has the single offset `[0]` and is valid only with an empty ID_001 topology.

The counts are explicit because ID_001 treats `tree_id` as a partition label and does not require tree IDs to be dense, sorted, or contiguous. ID_004 does not infer ragged boundaries from `tree_id`.

ID_004 consumes the ID_001 ABI as an already established topology contract. It checks the scalar node count needed for offset construction and does not duplicate topology validation of parent, depth, or tree identity.

For entry `b`, local node `l` is valid when `0 <= l < offset[b + 1] - offset[b]`, and its global packed index is `g = offset[b] + l`. The inverse lookup finds the unique non-empty entry whose half-open interval contains `g`. Empty entries own no global indices, so repeated boundaries are skipped by inverse lookup.
