#include "tree-draft-attention-baseline.h"

#include <algorithm>
#include <vector>

static size_t q_index_baseline(int32_t q, int32_t h, int32_t d, int32_t hq, int32_t dk) {
    return (static_cast<size_t>(q) * hq + h) * dk + d;
}

static size_t kv_index_baseline(size_t token, int32_t h, int32_t d, int32_t hkv, int32_t dim) {
    return (token * hkv + h) * dim + d;
}

common_tree_draft_attention_baseline_status common_tree_draft_attention_baseline_forward(
        const common_tree_draft_topology & topology,
        const common_tree_draft_ancestor_bitset & ancestors,
        const common_tree_draft_ragged_attention_batch & batch,
        const common_tree_draft_attention_oracle_view & view,
        const common_tree_draft_attention_launch & launch) {
    if (common_tree_draft_ragged_attention_validate(batch) != COMMON_TREE_DRAFT_RAGGED_ATTENTION_OK ||
        batch.n_tree_nodes != topology.n_nodes) {
        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_BATCH;
    }
    if (launch.block_k == 0 || (launch.num_warps != 1 && launch.num_warps != 2 &&
        launch.num_warps != 4 && launch.num_warps != 8)) {
        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_LAUNCH;
    }
    if (view.n_query_heads <= 0 || view.n_kv_heads <= 0 ||
        view.key_dim <= 0 || view.value_dim <= 0 ||
        view.output_count < static_cast<size_t>(topology.n_nodes) *
            static_cast<size_t>(view.n_query_heads) * static_cast<size_t>(view.value_dim)) {
        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_SHAPE;
    }
    if (view.q == nullptr || view.tree_k == nullptr || view.tree_v == nullptr || view.output == nullptr) {
        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_NULL_BUFFER;
    }

    common_tree_draft_head_map_plan heads;
    if (common_tree_draft_head_map_build(
            static_cast<uint32_t>(view.n_query_heads),
            static_cast<uint32_t>(view.n_kv_heads),
            {1,2,4,8,16,32,64}, &heads) != COMMON_TREE_DRAFT_HEAD_MAP_OK) {
        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_SHAPE;
    }

    const size_t words_per_row = common_tree_draft_ancestor_words_per_row(topology.n_nodes);
    common_tree_draft_ancestor_format format{
        ancestors.words,
        static_cast<uint32_t>(topology.n_nodes),
        static_cast<uint32_t>(words_per_row),
        COMMON_TREE_DRAFT_ANCESTOR_WORD_32,
    };
    if (common_tree_draft_ancestor_format_validate(format) != COMMON_TREE_DRAFT_ANCESTOR_FORMAT_OK) {
        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_BATCH;
    }

    size_t total_prefix = 0;
    for (int32_t e = 0; e < batch.n_entries; ++e) {
        total_prefix += static_cast<size_t>(batch.entries[e].prefix_len);
    }
    if (total_prefix > 0 && (view.prefix_k == nullptr || view.prefix_v == nullptr)) {
        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_NULL_BUFFER;
    }

    std::vector<size_t> prefix_offsets(static_cast<size_t>(batch.n_entries) + 1, 0);
    for (int32_t e = 0; e < batch.n_entries; ++e) {
        prefix_offsets[static_cast<size_t>(e) + 1] =
            prefix_offsets[static_cast<size_t>(e)] + static_cast<size_t>(batch.entries[e].prefix_len);
    }

    std::vector<float> accumulator(static_cast<size_t>(view.value_dim));
    std::vector<float> scores(static_cast<size_t>(launch.block_k));
    std::vector<float> values(static_cast<size_t>(launch.block_k) * static_cast<size_t>(view.value_dim));
    std::vector<uint8_t> masked(static_cast<size_t>(launch.block_k));

    auto score = [&](int32_t q_global, int32_t qh, int32_t kvh, const float * k, size_t token) {
        float dot = 0.0f;
        for (int32_t d = 0; d < view.key_dim; ++d) {
            dot += view.q[q_index_baseline(q_global,qh,d,view.n_query_heads,view.key_dim)] *
                   k[kv_index_baseline(token,kvh,d,view.n_kv_heads,view.key_dim)];
        }
        return dot * view.scale;
    };

    for (int32_t e = 0; e < batch.n_entries; ++e) {
        const auto & entry = batch.entries[e];
        for (int32_t q_local = 0; q_local < entry.q_len; ++q_local) {
            const int32_t q_global = entry.q_offset + q_local;
            for (int32_t qh = 0; qh < view.n_query_heads; ++qh) {
                const uint32_t kvh_u = common_tree_draft_head_map_kv_head(heads, static_cast<uint32_t>(qh));
                if (kvh_u == UINT32_MAX) return COMMON_TREE_DRAFT_ATTENTION_BASELINE_INVALID_SHAPE;
                const int32_t kvh = static_cast<int32_t>(kvh_u);

                common_tree_draft_online_softmax_state state;
                if (common_tree_draft_online_softmax_init(&state, accumulator.data(), accumulator.size()) !=
                    COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK) {
                    return COMMON_TREE_DRAFT_ATTENTION_BASELINE_SOFTMAX;
                }

                // Dense committed prefix tiles.
                for (int32_t base = 0; base < entry.prefix_len; base += static_cast<int32_t>(launch.block_k)) {
                    const int32_t rows = std::min<int32_t>(static_cast<int32_t>(launch.block_k), entry.prefix_len - base);
                    for (int32_t r = 0; r < rows; ++r) {
                        const size_t token = prefix_offsets[e] + static_cast<size_t>(base + r);
                        scores[static_cast<size_t>(r)] = score(q_global,qh,kvh,view.prefix_k,token);
                        masked[static_cast<size_t>(r)] = 0;
                        for (int32_t d = 0; d < view.value_dim; ++d) {
                            values[static_cast<size_t>(r) * view.value_dim + d] =
                                view.prefix_v[kv_index_baseline(token,kvh,d,view.n_kv_heads,view.value_dim)];
                        }
                    }
                    if (common_tree_draft_online_softmax_update(
                            &state,scores.data(),values.data(),masked.data(),static_cast<size_t>(rows)) !=
                        COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK) {
                        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_SOFTMAX;
                    }
                }

                // Proposal-tree tiles with O(1) ancestry decode and safe tails.
                for (int32_t base = 0; base < entry.tree_len; base += static_cast<int32_t>(launch.block_k)) {
                    const int32_t rows = std::min<int32_t>(static_cast<int32_t>(launch.block_k), entry.tree_len - base);
                    for (int32_t r = 0; r < rows; ++r) {
                        const int32_t k_global = entry.tree_offset + base + r;
                        const bool same_tree = topology.tree_id[q_global] == topology.tree_id[k_global];
                        const bool visible = same_tree &&
                            common_tree_draft_visibility_decode(
                                format, static_cast<uint32_t>(q_global), static_cast<uint32_t>(k_global));
                        masked[static_cast<size_t>(r)] = static_cast<uint8_t>(!visible);
                        scores[static_cast<size_t>(r)] = visible ?
                            score(q_global,qh,kvh,view.tree_k,static_cast<size_t>(k_global)) : 0.0f;
                        for (int32_t d = 0; d < view.value_dim; ++d) {
                            values[static_cast<size_t>(r) * view.value_dim + d] = visible ?
                                view.tree_v[kv_index_baseline(static_cast<size_t>(k_global),kvh,d,view.n_kv_heads,view.value_dim)] : 0.0f;
                        }
                    }
                    if (common_tree_draft_online_softmax_update(
                            &state,scores.data(),values.data(),masked.data(),static_cast<size_t>(rows)) !=
                        COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK) {
                        return COMMON_TREE_DRAFT_ATTENTION_BASELINE_SOFTMAX;
                    }
                }

                float * out = view.output +
                    (static_cast<size_t>(q_global) * view.n_query_heads + qh) * view.value_dim;
                if (common_tree_draft_online_softmax_finalize(state,out,static_cast<size_t>(view.value_dim)) !=
                    COMMON_TREE_DRAFT_ONLINE_SOFTMAX_OK) {
                    return COMMON_TREE_DRAFT_ATTENTION_BASELINE_SOFTMAX;
                }
            }
        }
    }
    return COMMON_TREE_DRAFT_ATTENTION_BASELINE_OK;
}
