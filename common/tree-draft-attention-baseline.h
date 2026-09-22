#pragma once

#include "tree-draft-attention-oracle.h"
#include "tree-draft-head-map.h"
#include "tree-draft-online-softmax.h"
#include "tree-draft-ragged-attention.h"
#include "tree-draft-visibility-decode.h"

#include <cstddef>
#include <cstdint>

struct common_tree_draft_attention_launch {
    uint32_t block_k = 32;
    uint32_t num_warps = 4;
};

enum common_tree_draft_attention_baseline_status : uint32_t {
    COMMON_TREE_DRAFT_ATTENTION_BASELINE_OK = 0,
    COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_BATCH,
    COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_SHAPE,
    COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_LAUNCH,
    COMMON_TREE_DRAFT_ATTENTION_BASELINE_NULL_BUFFER,
    COMMON_TREE_DRAFT_ATTENTION_BASELINE_SOFTMAX,
};

common_tree_draft_attention_baseline_status common_tree_draft_attention_baseline_forward(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        const common_tree_draft_ragged_attention_batch & batch,
        const common_tree_draft_attention_oracle_view & view,
        const common_tree_draft_attention_launch & launch);
