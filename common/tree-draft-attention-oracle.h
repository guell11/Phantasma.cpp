#pragma once

#include "tree-draft-composite-mask.h"

#include <cstddef>
#include <cstdint>

// Slow scalar fp32 reference for committed-prefix + proposal-tree attention.
// Q is packed by global proposal query order [Q,Hq,Dk].
// Prefix K/V are concatenated by ragged entry [sum(P_b),Hkv,D].
// Tree K/V are packed by global proposal node order [T,Hkv,D].
// Output is [Q,Hq,Dv].
struct common_tree_draft_attention_oracle_view {
    const float * q = nullptr;
    const float * prefix_k = nullptr;
    const float * prefix_v = nullptr;
    const float * tree_k = nullptr;
    const float * tree_v = nullptr;
    float * output = nullptr;

    int32_t n_query_heads = 0;
    int32_t n_kv_heads = 0;
    int32_t key_dim = 0;
    int32_t value_dim = 0;
    float scale = 0.0f;
    size_t output_count = 0;
};

enum common_tree_draft_attention_oracle_status : uint32_t {
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_OK = 0,
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_FOREST,
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_PREFIX,
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_SHAPE,
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_NULL_BUFFER,
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_OUTPUT_TOO_SMALL,
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_NUMERIC,
    COMMON_TREE_DRAFT_ATTENTION_ORACLE_MASK,
};

common_tree_draft_attention_oracle_status common_tree_draft_attention_oracle(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        const common_tree_draft_attention_oracle_view & view);
