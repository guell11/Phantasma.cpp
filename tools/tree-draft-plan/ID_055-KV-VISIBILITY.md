# ID_055 tree-KV ancestry visibility

ID_055 derives constant-time visibility metadata from the accepted ID_054 branch descriptors. The chosen representation is a dense branch-by-branch cutoff table rebuilt at draft-tree mutation boundaries. Entry cutoff[q,k] is the greatest logical position on key branch k that remains on query branch q's lineage, or -1 when k is not an ancestor branch of q.

For a branch itself, the cutoff is its tip position. For a child branch, the direct parent cutoff is the child's fork position. More distant ancestor cutoffs are copied from the parent row and clipped by the child's fork position. This preserves the ID_002 inclusive ancestor relation without walking parent pointers during token visibility checks.

A key at (key_branch,key_position) is visible to a query at (query_branch,query_position) exactly when key_position <= query_position and key_position <= cutoff[query_branch,key_branch]. Siblings and descendants therefore remain invisible, while shared ancestry stays visible only through the earliest fork on the path.

The table uses caller-owned contiguous int64_t storage and canonical branch indices, so one immutable snapshot can be uploaded as metadata for later Tree PagedAttention modules. Branch handles include generation, parent descriptors must precede children, and stale-generation parent handles are rejected instead of aliasing a recycled branch id.

DFS intervals were not selected because dynamic branch insertion changes interval boundaries and still does not encode the fork-position cutoff needed for shared-prefix visibility. The cutoff table costs O(B^2) metadata, builds each row from an already materialized parent row, and gives the required O(1) predicate for the bounded speculative branch sets used by the tree scheduler.

