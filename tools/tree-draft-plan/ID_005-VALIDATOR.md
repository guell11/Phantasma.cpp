# ID_005 topology validator

`common_tree_draft_validate()` is the canonical preflight for packed Tree-Draft topology metadata. It consumes the accepted ID_001 `common_tree_draft_topology` and ID_004 `common_tree_draft_forest_offsets` contracts.

The validator first runs the complete ID_001 topology validator. This enforces non-null buffers for non-empty inputs, non-negative tree IDs, root depth zero, parent-before-child ordering, parent/tree agreement, exact depth recurrence, and one root per tree ID. Because every non-root parent is strictly earlier than its child, a valid topology cannot contain a parent cycle.

The forest descriptor must then have a non-negative entry count and a non-null offset buffer. `offset[0]` is exactly zero, offsets are non-negative and non-decreasing, every offset is at most `topology.n_nodes`, and `offset[n_entries]` is exactly `topology.n_nodes`. Repeated offsets are valid empty entries.

Malformed-input checks are always on. None of the parent, depth, tree identity, or offset checks is debug-only because downstream kernels use these values for indexing. Debug builds may add assertions around internal callers, but those assertions do not define extra accepted inputs or alternate failure semantics.

`COMMON_TREE_DRAFT_VALIDATOR_INVALID_TOPOLOGY` preserves the exact ID_001 cause in `common_tree_draft_validation_result::topology_error`. Forest failures have stable validator error names. Validation does not mutate topology or offset storage.
