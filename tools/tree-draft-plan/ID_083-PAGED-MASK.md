# ID_083 paged tree-attention mask semantics

ID_083 treats each ID_081 gather row as the complete ordered key set for one packed query. The gather builder has already applied the ID_055 ancestry cutoff to the selected branch path, so a key is visible only when its segment belongs to the query row and its page offset lies inside that segment's half-open range.

Logical key positions are reconstructed from the concatenated segment lengths, beginning at position zero for each query row. The first gathered token therefore retains BOS/root position zero, and the last token is causal with respect to the query position established by ID_081. Equal logical positions in different packed query rows remain independent; row boundaries prevent sibling or cross-branch leakage even when pages are physically shared.

The CPU predicate returns a logical position only for a visible key. Keys from another query row are masked rather than reinterpreted through physical page identity. Invalid segment offsets, malformed CSR metadata, and arithmetic overflow are reported explicitly.

Additive mask values reuse the accepted ID_010 convention exactly: visible edges contribute 0 and disallowed edges contribute negative infinity before softmax. ID_084 owns sliding-window clipping; it can use the logical positions exposed here without changing ancestry or causal semantics.
