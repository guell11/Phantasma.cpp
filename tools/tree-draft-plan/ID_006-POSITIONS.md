# ID_006 tree position mapping

`common_tree_draft_positions_build()` consumes the accepted ID_001 packed topology and ID_004 packed forest offsets. It produces one contiguous `int32_t` position per global packed node, matching `llama_pos` width.

For packed entry `b`, prefix length `P_b`, and global node `g` in the half-open interval `[offset[b], offset[b + 1])`:

```text
position[g] = P_b + depth[g]
```

Prefix lengths are supplied as one non-negative `int32_t` value per packed entry, so different entries may use different committed-prefix lengths. Tree depth, not local node order, determines position. Siblings therefore intentionally share the same position.

ID_004 owns the entry/local/global mapping. For valid `entry=b` and `local=l`, `global=offset[b]+l`; `common_tree_draft_position_for_local()` reuses that mapping and returns the same `position[global]` value the packed builder produces. Repeated offsets represent empty entries and own no local/global nodes.

Before writing any output, the builder validates the complete ID_001 topology, the ID_004 offset shape (`offset[0]=0`, monotonic non-negative boundaries, final offset equal to `topology.n_nodes`), every prefix length, output capacity, and every `P_b + depth[g]` addition. A non-root parent must lie in the same packed entry as its child; splitting one tree across entries would apply different prefix origins to one ancestry chain and is rejected as `invalid_forest`. Any sum above `INT32_MAX` returns `overflow` without partially writing the position buffer.

For two non-empty entries separated by an empty entry:

```text
offsets = [0, 3, 3, 6]
prefix  = [100, 777, 2000]
depth   = [0, 1, 1, 0, 1, 1]
position= [100, 101, 101, 2000, 2001, 2001]
```

The empty entry's prefix is valid metadata but contributes no output position. ID_006 does not modify KV state, attention masks, or target verification behavior.
