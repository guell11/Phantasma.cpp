# ID_008 tile-level tree visibility

ID_008 consumes the accepted ID_002 ancestor bitset and ID_004 packed-forest offsets. It defines a conservative tile predicate used only to reject query-key tiles that are provably fully invisible before any K/V load.

For logical packed nodes `i` and `j`, the exact element predicate is:

```text
visible(i,j) = same_packed_entry(i,j) && ancestor_contains(i,j)
```

The packed-entry check comes from the ID_004 half-open ranges. Empty entries own no nodes. This keeps ragged batches independent even when one global tile straddles an entry boundary.

For global query tile `Q_t` and key tile `K_t`, clipped to `[0,N)` at the tail:

```text
tile_active(Q_t,K_t) = OR visible(i,j)
                       over i in Q_t and j in K_t
```

The CPU/reference builder computes this predicate exactly. An accelerator may replace `tile_active` with any cheaper conservative over-approximation, but it must never produce a false negative: a tile may be marked active even when all elements are invisible, but it may be rejected only when every exact element predicate in the tile is false.

Tile metadata is a row-major `uint8_t[n_query_tiles][n_key_tiles]` array where zero means provably inactive and nonzero means potentially active. `n_query_tiles=ceil(N/query_tile_size)` and `n_key_tiles=ceil(N/key_tile_size)`. Tail tiles are clipped rather than padded with logical nodes.

Element visibility remains authoritative inside every active tile. Marking an entire tile active does not make siblings, descendants-to-ancestors, or nodes from different packed entries visible. A kernel using conservative tile metadata must still apply the exact ID_002 ancestry predicate for element masking.

ID_008 does not define KV indirection, position masking, sparse scheduling, or attention kernels. It only defines the tile rejection boundary and its CPU oracle.
