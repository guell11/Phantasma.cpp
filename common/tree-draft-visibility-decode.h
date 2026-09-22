#pragma once

#include "tree-draft-ancestor-format.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_visibility_lane {
    uint32_t key = 0;
    uint8_t visible = 0;
};

// Scalar O(1) decode used as the exact C++ oracle for Triton/vector lanes.
// Any out-of-range node/key is invisible by definition.
bool common_tree_draft_visibility_decode(
        const common_tree_draft_ancestor_format & format,
        uint32_t query_node,
        uint32_t key_node);

// Safe-tail vector decode: every lane is written exactly once; out-of-range
// keys become visible=0 without touching storage outside the validated bitset.
void common_tree_draft_visibility_decode_lanes(
        const common_tree_draft_ancestor_format & format,
        uint32_t query_node,
        common_tree_draft_visibility_lane * lanes,
        size_t lane_count);
