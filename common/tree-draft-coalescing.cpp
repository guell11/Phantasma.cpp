#include "tree-draft-coalescing.h"

#include <algorithm>
#include <limits>
#include <vector>

common_tree_draft_coalescing_status common_tree_draft_coalescing_measure(
        const common_tree_draft_lane_access * lanes,
        size_t lane_count,
        common_tree_draft_coalescing_result * result) {
    if (result == nullptr || (lane_count > 0 && lanes == nullptr)) {
        return COMMON_TREE_DRAFT_COALESCING_NULL_INPUT;
    }
    if (lane_count == 0) return COMMON_TREE_DRAFT_COALESCING_EMPTY;

    std::vector<uint64_t> sectors;
    uint64_t useful = 0;
    bool contiguous = true;
    for (size_t i = 0; i < lane_count; ++i) {
        if (lanes[i].useful_bytes == 0) {
            contiguous = false;
            continue;
        }
        if (useful > std::numeric_limits<uint64_t>::max() - lanes[i].useful_bytes ||
            lanes[i].address > std::numeric_limits<uint64_t>::max() - (lanes[i].useful_bytes - 1)) {
            return COMMON_TREE_DRAFT_COALESCING_OVERFLOW;
        }
        useful += lanes[i].useful_bytes;
        const uint64_t first = lanes[i].address / 32;
        const uint64_t last = (lanes[i].address + lanes[i].useful_bytes - 1) / 32;
        for (uint64_t s = first; s <= last; ++s) sectors.push_back(s);
        if (i > 0) {
            const uint64_t expected = lanes[i - 1].address + lanes[i - 1].useful_bytes;
            if (lanes[i].address != expected) contiguous = false;
        }
    }
    std::sort(sectors.begin(), sectors.end());
    sectors.erase(std::unique(sectors.begin(), sectors.end()), sectors.end());
    if (sectors.empty()) return COMMON_TREE_DRAFT_COALESCING_EMPTY;

    common_tree_draft_coalescing_result out;
    out.sectors_touched = static_cast<uint32_t>(sectors.size());
    out.useful_bytes = useful;
    out.efficiency = static_cast<float>(
        static_cast<double>(useful) / (32.0 * static_cast<double>(sectors.size())));
    out.adjacent_lanes_contiguous = contiguous;
    *result = out;
    return COMMON_TREE_DRAFT_COALESCING_OK;
}
