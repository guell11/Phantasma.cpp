# ID_085 hybrid memory integration boundary

ID_085 classifies each layer memory use as attention KV or recurrent state. Attention layers consume the paged ancestry gather metadata established by ID_083. Recurrent layers never receive paged KV metadata; they carry explicit sequence and speculative-branch identity instead.

The binding contract rejects a recurrent layer when paged KV metadata is supplied. A recurrent binding requires both sequence and branch identity and advertises that speculative execution needs state fork and rollback support. This keeps recurrent state lifetime separate from token KV page sharing.

The boundary is metadata-only. It does not implement recurrent kernels, state copying, or rollback mechanics. Callers map model-specific per-layer recurrent metadata, such as the existing hybrid-layer flags, to the memory mode before binding a Tree-Draft layer.
