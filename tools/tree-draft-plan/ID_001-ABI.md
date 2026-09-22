# ID_001 packed tree topology ABI

The host representation is `common_tree_draft_topology` from `common/tree-draft-topology.h`.

The logical ABI is four kernel arguments in this order:

1. `parent`: pointer to contiguous `int32`, shape `[n_nodes]`.
2. `depth`: pointer to contiguous `int32`, shape `[n_nodes]`.
3. `tree_id`: pointer to contiguous `int32`, shape `[n_nodes]`.
4. `n_nodes`: scalar `int32` in `[0, INT32_MAX]`.

C++ uses `int32_t`; Triton consumers use `tl.int32`. Buffer strides are one element. The same node index addresses all three buffers. No padding, structure-of-arrays reinterpretation, or implicit permutation is part of the ABI.

Nodes are packed in topological order. A root has `parent[i] == -1` and `depth[i] == 0`. Every non-root has `0 <= parent[i] < i`, belongs to the same `tree_id` as its parent, and has `depth[i] == depth[parent[i]] + 1`.

Tree IDs are non-negative partition labels. They need not be dense, sorted, or contiguous in packed order. Every distinct tree ID has exactly one root, and every non-root has the same tree ID as its parent. A Triton consumer can therefore reject cross-tree nodes with one `tree_id` comparison without assigning meaning to the numeric ID itself.

An empty forest has `n_nodes == 0`; its three pointers may be null. For a non-empty forest all three pointers must be non-null. Negative tree IDs, negative node counts, invalid parents, root depth mismatches, depth recurrence mismatches, parent/tree mismatches, and repeated roots for a tree are validation errors. `common_tree_draft_topology_error_name()` provides stable lowercase diagnostic names.

Example with two trees:

```text
index    0  1  2  3  4  5
parent  -1  0  0  1 -1  4
depth    0  1  1  2  0  1
tree_id  0  0  0  0  1  1
```

This represents tree 0 with root 0, children 1 and 2, and grandchild 3 under node 1, followed by tree 1 with root 4 and child 5.
