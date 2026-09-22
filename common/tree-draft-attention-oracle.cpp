#include "tree-draft-attention-oracle.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

static size_t q_index(int32_t q, int32_t h, int32_t d, int32_t hq, int32_t dk) {
    return (static_cast<size_t>(q) * hq + h) * dk + d;
}

static size_t kv_index(size_t token, int32_t h, int32_t d, int32_t hkv, int32_t dim) {
    return (token * hkv + h) * dim + d;
}

common_tree_draft_attention_oracle_status common_tree_draft_attention_oracle(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        const common_tree_draft_forest_offsets & forest,
        const int32_t * prefix_lengths,
        const common_tree_draft_attention_oracle_view & view) {
    if (common_tree_draft_topology_validate(topology) != COMMON_TREE_DRAFT_TOPOLOGY_OK) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_TOPOLOGY;
    }
    if (forest.n_entries < 0 || forest.offsets == nullptr || forest.offsets[0] != 0) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_FOREST;
    }
    if (prefix_lengths == nullptr) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_PREFIX;
    }
    for (int32_t e = 0; e < forest.n_entries; ++e) {
        if (forest.offsets[e] < 0 || forest.offsets[e + 1] < forest.offsets[e]) {
            return COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_FOREST;
        }
        if (prefix_lengths[e] < 0) {
            return COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_PREFIX;
        }
    }
    if (forest.offsets[forest.n_entries] != topology.n_nodes) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_FOREST;
    }
    if (view.n_query_heads <= 0 || view.n_kv_heads <= 0 ||
        view.n_query_heads % view.n_kv_heads != 0 ||
        view.key_dim <= 0 || view.value_dim <= 0 ||
        !std::isfinite(view.scale)) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_INVALID_SHAPE;
    }
    if (topology.n_nodes == 0) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_OK;
    }
    if (view.q == nullptr || view.tree_k == nullptr || view.tree_v == nullptr || view.output == nullptr) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_NULL_BUFFER;
    }

    size_t total_prefix = 0;
    for (int32_t e = 0; e < forest.n_entries; ++e) {
        total_prefix += static_cast<size_t>(prefix_lengths[e]);
    }
    if (total_prefix > 0 && (view.prefix_k == nullptr || view.prefix_v == nullptr)) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_NULL_BUFFER;
    }
    const size_t needed_output = static_cast<size_t>(topology.n_nodes) *
        static_cast<size_t>(view.n_query_heads) * static_cast<size_t>(view.value_dim);
    if (view.output_count < needed_output) {
        return COMMON_TREE_DRAFT_ATTENTION_ORACLE_OUTPUT_TOO_SMALL;
    }

    std::vector<size_t> prefix_offsets(static_cast<size_t>(forest.n_entries) + 1, 0);
    for (int32_t e = 0; e < forest.n_entries; ++e) {
        prefix_offsets[static_cast<size_t>(e) + 1] =
            prefix_offsets[static_cast<size_t>(e)] + static_cast<size_t>(prefix_lengths[e]);
    }
    const int32_t group = view.n_query_heads / view.n_kv_heads;
    std::vector<float> accumulator(static_cast<size_t>(view.value_dim));

    for (int32_t e = 0; e < forest.n_entries; ++e) {
        const int32_t q_begin = forest.offsets[e];
        const int32_t q_end = forest.offsets[e + 1];
        const int32_t tree_len = q_end - q_begin;
        for (int32_t q_local = 0; q_local < tree_len; ++q_local) {
            const int32_t q_global = q_begin + q_local;
            for (int32_t qh = 0; qh < view.n_query_heads; ++qh) {
                const int32_t kvh = qh / group;
                float row_max = -std::numeric_limits<float>::infinity();

                auto dot_score = [&](const float * k, size_t token) -> float {
                    float dot = 0.0f;
                    for (int32_t d = 0; d < view.key_dim; ++d) {
                        dot += view.q[q_index(q_global, qh, d, view.n_query_heads, view.key_dim)] *
                               k[kv_index(token, kvh, d, view.n_kv_heads, view.key_dim)];
                    }
                    return dot * view.scale;
                };

                for (int32_t p = 0; p < prefix_lengths[e]; ++p) {
                    bool visible = false;
                    const auto ms = common_tree_draft_composite_mask_visible(
                        topology, ancestors, forest, prefix_lengths, e, q_local,
                        COMMON_TREE_DRAFT_COMPOSITE_KEY_PREFIX, p, &visible);
                    if (ms != COMMON_TREE_DRAFT_COMPOSITE_MASK_OK) return COMMON_TREE_DRAFT_ATTENTION_ORACLE_MASK;
                    if (!visible) continue;
                    const float s = dot_score(view.prefix_k, prefix_offsets[e] + static_cast<size_t>(p));
                    if (!std::isfinite(s)) return COMMON_TREE_DRAFT_ATTENTION_ORACLE_NUMERIC;
                    row_max = std::max(row_max, s);
                }
                for (int32_t k_local = 0; k_local < tree_len; ++k_local) {
                    bool visible = false;
                    const auto ms = common_tree_draft_composite_mask_visible(
                        topology, ancestors, forest, prefix_lengths, e, q_local,
                        COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, k_local, &visible);
                    if (ms != COMMON_TREE_DRAFT_COMPOSITE_MASK_OK) return COMMON_TREE_DRAFT_ATTENTION_ORACLE_MASK;
                    if (!visible) continue;
                    const float s = dot_score(view.tree_k, static_cast<size_t>(q_begin + k_local));
                    if (!std::isfinite(s)) return COMMON_TREE_DRAFT_ATTENTION_ORACLE_NUMERIC;
                    row_max = std::max(row_max, s);
                }
                if (!std::isfinite(row_max)) return COMMON_TREE_DRAFT_ATTENTION_ORACLE_NUMERIC;

                std::fill(accumulator.begin(), accumulator.end(), 0.0f);
                float denom = 0.0f;
                auto accumulate = [&](float score, const float * v, size_t token) {
                    const float w = std::exp(score - row_max);
                    denom += w;
                    for (int32_t d = 0; d < view.value_dim; ++d) {
                        accumulator[static_cast<size_t>(d)] +=
                            w * v[kv_index(token, kvh, d, view.n_kv_heads, view.value_dim)];
                    }
                };
                for (int32_t p = 0; p < prefix_lengths[e]; ++p) {
                    const size_t token = prefix_offsets[e] + static_cast<size_t>(p);
                    accumulate(dot_score(view.prefix_k, token), view.prefix_v, token);
                }
                for (int32_t k_local = 0; k_local < tree_len; ++k_local) {
                    bool visible = false;
                    const auto ms = common_tree_draft_composite_mask_visible(
                        topology, ancestors, forest, prefix_lengths, e, q_local,
                        COMMON_TREE_DRAFT_COMPOSITE_KEY_PROPOSAL, k_local, &visible);
                    if (ms != COMMON_TREE_DRAFT_COMPOSITE_MASK_OK) return COMMON_TREE_DRAFT_ATTENTION_ORACLE_MASK;
                    if (!visible) continue;
                    const size_t token = static_cast<size_t>(q_begin + k_local);
                    accumulate(dot_score(view.tree_k, token), view.tree_v, token);
                }
                if (!(denom > 0.0f) || !std::isfinite(denom)) return COMMON_TREE_DRAFT_ATTENTION_ORACLE_NUMERIC;
                for (int32_t d = 0; d < view.value_dim; ++d) {
                    const float out = accumulator[static_cast<size_t>(d)] / denom;
                    if (!std::isfinite(out)) return COMMON_TREE_DRAFT_ATTENTION_ORACLE_NUMERIC;
                    view.output[(static_cast<size_t>(q_global) * view.n_query_heads + qh) * view.value_dim + d] = out;
                }
            }
        }
    }
    return COMMON_TREE_DRAFT_ATTENTION_ORACLE_OK;
}
