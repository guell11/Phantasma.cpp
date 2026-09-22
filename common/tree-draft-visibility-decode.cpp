#include "tree-draft-visibility-decode.h"

bool common_tree_draft_visibility_decode(
        const common_tree_draft_ancestor_format & format,
        uint32_t query_node,
        uint32_t key_node) {
    if (query_node >= format.n_nodes || key_node >= format.n_nodes) return false;
    return common_tree_draft_ancestor_format_contains(format, query_node, key_node);
}

void common_tree_draft_visibility_decode_lanes(
        const common_tree_draft_ancestor_format & format,
        uint32_t query_node,
        common_tree_draft_visibility_lane * lanes,
        size_t lane_count) {
    if (lanes == nullptr) return;
    for (size_t i = 0; i < lane_count; ++i) {
        lanes[i].visible = static_cast<uint8_t>(common_tree_draft_visibility_decode(format, query_node, lanes[i].key));
    }
}
