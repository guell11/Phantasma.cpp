# ID_084 sliding-window paged visibility

ID_084 consumes the complete ancestry-and-causal gather rows defined by ID_083 and derives a layer-specific segment view for sliding-window attention. The source gather metadata stays unchanged so layers with different windows can reuse the same ancestry gather.

For query position q with window start s, the visible logical interval is [max(s, 0), q + 1). Each source segment already represents a contiguous logical interval in row order. Clipping intersects that logical interval with the visible interval and adjusts only the segment lo/hi offsets; page identity and generation remain unchanged.

The clipper validates that each source row spans exactly q + 1 logical positions. This preserves the ID_083 position mapping and rejects stale or mismatched query metadata. A window start beyond q is rejected because it would remove the query token itself and leave no valid attention domain.

Window parameters are supplied per query invocation. A caller may therefore run the same source gather through different per-layer window starts without rebuilding ancestry metadata or changing branch visibility semantics.
