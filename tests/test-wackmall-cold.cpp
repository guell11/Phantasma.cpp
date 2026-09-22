#include "ggml.h"
#include "ggml-cpu.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static std::atomic<int> g_fetch_calls[4];

static const char * count_fetch(const void *, int64_t expert, const char * fallback) {
    if (expert >= 0 && expert < 4) {
        g_fetch_calls[expert].fetch_add(1, std::memory_order_relaxed);
    }
    return fallback;
}

static void reset_fetch_calls() {
    for (auto & count : g_fetch_calls) {
        count.store(0, std::memory_order_relaxed);
    }
}

static bool nearly_equal(float a, float b) {
    const float scale = 1.0f + std::fabs(a) + std::fabs(b);
    return std::fabs(a - b) <= 1e-5f * scale;
}

static int run_fused_case(int64_t n_tokens) {
    const int64_t n_expert = 4;
    const int64_t n_used = 2;
    const int64_t n_embd = 4;
    const int64_t n_ff = 3;

    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "failed to create fused ggml context\n");
        return 1;
    }

    ggml_tensor * gate   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * up     = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, n_ff, n_expert);
    ggml_tensor * down   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_ff, n_embd, n_expert);
    ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, 1, n_tokens);
    ggml_tensor * ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * mask   = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert);
    ggml_tensor * counts = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert + 1);

    float * wg = static_cast<float *>(gate->data);
    float * wu = static_cast<float *>(up->data);
    float * wd = static_cast<float *>(down->data);
    for (int64_t e = 0; e < n_expert; ++e) {
        for (int64_t row = 0; row < n_ff; ++row) {
            for (int64_t col = 0; col < n_embd; ++col) {
                const size_t i = (size_t) col + (size_t) n_embd * ((size_t) row + (size_t) n_ff * (size_t) e);
                wg[i] = 0.02f * float(1 + col) + 0.03f * float(1 + row) + 0.05f * float(e);
                wu[i] = 0.04f * float(1 + col) - 0.01f * float(1 + row) + 0.02f * float(e);
            }
        }
        for (int64_t row = 0; row < n_embd; ++row) {
            for (int64_t col = 0; col < n_ff; ++col) {
                const size_t i = (size_t) col + (size_t) n_ff * ((size_t) row + (size_t) n_embd * (size_t) e);
                wd[i] = 0.03f * float(1 + col) + 0.02f * float(1 + row) - 0.01f * float(e);
            }
        }
    }

    float * x = static_cast<float *>(input->data);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t col = 0; col < n_embd; ++col) {
            x[col + n_embd*t] = 0.05f * float(1 + col) + 0.01f * float(t);
        }
    }

    std::vector<int32_t> route((size_t) n_used*n_tokens);
    std::vector<int32_t> expected_counts((size_t) n_expert + 1, 0);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t slot = 0; slot < n_used; ++slot) {
            const int32_t expert = (int32_t) ((3*t + slot) % n_expert);
            route[(size_t) slot + (size_t) n_used*t] = expert;
            expected_counts[(size_t) expert]++;
            expected_counts[(size_t) n_expert]++;
        }
    }
    std::memcpy(ids->data, route.data(), route.size()*sizeof(route[0]));
    int32_t * mask_data = static_cast<int32_t *>(mask->data);
    mask_data[0] = 1;
    mask_data[1] = 0;
    mask_data[2] = 1;
    mask_data[3] = 1;
    std::memset(counts->data, 0, ggml_nbytes(counts));

    ggml_tensor * fused = ggml_moe_cold(ctx, gate, up, down, input, ids, mask, counts, nullptr, nullptr, nullptr);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, fused);
    const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 4);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "fused graph compute failed: %s\n", ggml_status_to_string(status));
        ggml_free(ctx);
        return 1;
    }

    const float * got = static_cast<const float *>(fused->data);
    std::vector<float> act((size_t) n_ff);
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float * xt = x + n_embd*t;
        for (int64_t slot = 0; slot < n_used; ++slot) {
            const int32_t e = route[(size_t) slot + (size_t) n_used*t];
            for (int64_t j = 0; j < n_ff; ++j) {
                float g = 0.0f;
                float u = 0.0f;
                for (int64_t i = 0; i < n_embd; ++i) {
                    const size_t wi = (size_t) i + (size_t) n_embd * ((size_t) j + (size_t) n_ff * (size_t) e);
                    g += wg[wi] * xt[i];
                    u += wu[wi] * xt[i];
                }
                act[(size_t) j] = (g / (1.0f + std::exp(-g))) * u;
            }
            for (int64_t row = 0; row < n_embd; ++row) {
                float expected = 0.0f;
                if (mask_data[e]) {
                    for (int64_t j = 0; j < n_ff; ++j) {
                        const size_t wi = (size_t) j + (size_t) n_ff * ((size_t) row + (size_t) n_embd * (size_t) e);
                        expected += wd[wi] * act[(size_t) j];
                    }
                }
                const size_t oi = (size_t) row + (size_t) n_embd * ((size_t) slot + (size_t) n_used * (size_t) t);
                if (!nearly_equal(got[oi], expected)) {
                    std::fprintf(stderr,
                            "fused cold mismatch t=%lld slot=%lld expert=%d row=%lld got=%g expected=%g\n",
                            (long long) t, (long long) slot, e, (long long) row, got[oi], expected);
                    ggml_free(ctx);
                    return 1;
                }
            }
        }
    }

    const int32_t * count_data = static_cast<const int32_t *>(counts->data);
    for (int i = 0; i < n_expert + 1; ++i) {
        if (count_data[i] != expected_counts[(size_t) i]) {
            std::fprintf(stderr, "fused count mismatch i=%d got=%d expected=%d\n",
                    i, count_data[i], expected_counts[(size_t) i]);
            ggml_free(ctx);
            return 1;
        }
    }

    ggml_free(ctx);
    return 0;
}

static int run_case(int64_t n_tokens) {
    const int64_t n_expert = 4;
    const int64_t n_used = 2;
    const int64_t k = 4;
    const int64_t m = 3;

    ggml_init_params params = {
        /* .mem_size   = */ 16 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ false,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::fprintf(stderr, "failed to create ggml context\n");
        return 1;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_expert);
    ggml_tensor * input   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, n_tokens);
    ggml_tensor * ids     = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * mask    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert);
    ggml_tensor * counts  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert + 1);
    ggml_tensor * ptrs    = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_expert);

    float * w = static_cast<float *>(weights->data);
    for (int64_t e = 0; e < n_expert; ++e) {
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < k; ++col) {
                w[col + k * (row + m * e)] =
                    0.01f * float(1 + col) + 0.1f * float(1 + row) + float(e);
            }
        }
    }

    float * x = static_cast<float *>(input->data);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t slot = 0; slot < n_used; ++slot) {
            for (int64_t col = 0; col < k; ++col) {
                x[col + k * (slot + n_used * t)] =
                    0.05f * float(1 + col) + 0.2f * float(slot) + 0.3f * float(t);
            }
        }
    }

    std::vector<int32_t> route((size_t) n_used*n_tokens);
    std::vector<int32_t> expected_counts((size_t) n_expert + 1, 0);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t slot = 0; slot < n_used; ++slot) {
            const int32_t expert = (int32_t) ((t + slot) % n_expert);
            route[(size_t) slot + (size_t) n_used*t] = expert;
            expected_counts[(size_t) expert]++;
            expected_counts[(size_t) n_expert]++;
        }
    }
    std::memcpy(ids->data, route.data(), route.size()*sizeof(route[0]));
    std::memset(counts->data, 0, ggml_nbytes(counts));
    int64_t * ptr_data = static_cast<int64_t *>(ptrs->data);
    const size_t expert_bytes = ggml_nbytes(weights)/(size_t) n_expert;
    for (int64_t e = 0; e < n_expert; ++e) {
        ptr_data[e] = (int64_t) (uintptr_t) ((const char *) weights->data + (size_t) e*expert_bytes);
    }

    ggml_tensor * stock = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_tensor * cold  = ggml_mul_mat_id_cold(ctx, weights, input, ids, mask, ptrs);
    ggml_tensor * count = ggml_moe_count(ctx, ids, counts);

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, stock);
    ggml_build_forward_expand(graph, cold);
    ggml_build_forward_expand(graph, count);

    auto compute = [&]() {
        const ggml_status status = ggml_graph_compute_with_ctx(ctx, graph, 4);
        if (status != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "graph compute failed: %s\n", ggml_status_to_string(status));
            return false;
        }
        return true;
    };

    int32_t * mask_data = static_cast<int32_t *>(mask->data);
    mask_data[0] = 1;
    mask_data[1] = 0;
    mask_data[2] = 1;
    mask_data[3] = 0;
    std::memset(counts->data, 0, ggml_nbytes(counts));
    reset_fetch_calls();

    if (!compute()) {
        ggml_free(ctx);
        return 1;
    }

    const float * stock_data = static_cast<const float *>(stock->data);
    const float * cold_data  = static_cast<const float *>(cold->data);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t slot = 0; slot < n_used; ++slot) {
            const int32_t expert = route[(size_t) slot + (size_t) n_used*t];
            for (int64_t row = 0; row < m; ++row) {
                const int64_t index = row + m * (slot + n_used * t);
                const float expected = mask_data[expert] ? stock_data[index] : 0.0f;
                if (!nearly_equal(cold_data[index], expected)) {
                    std::fprintf(stderr,
                            "mixed mask mismatch t=%lld slot=%lld expert=%d row=%lld got=%g expected=%g\n",
                            (long long) t, (long long) slot, expert, (long long) row,
                            cold_data[index], expected);
                    ggml_free(ctx);
                    return 1;
                }
            }
        }
    }
    if (g_fetch_calls[0].load(std::memory_order_relaxed) != 1 ||
            g_fetch_calls[1].load(std::memory_order_relaxed) != 0 ||
            g_fetch_calls[2].load(std::memory_order_relaxed) != 1 ||
            g_fetch_calls[3].load(std::memory_order_relaxed) != 0) {
        std::fprintf(stderr, "mixed mask fetch-hook calls mismatch: %d,%d,%d,%d\n",
                g_fetch_calls[0].load(std::memory_order_relaxed),
                g_fetch_calls[1].load(std::memory_order_relaxed),
                g_fetch_calls[2].load(std::memory_order_relaxed),
                g_fetch_calls[3].load(std::memory_order_relaxed));
        ggml_free(ctx);
        return 1;
    }

    const int32_t * count_data = static_cast<const int32_t *>(counts->data);
    for (int i = 0; i < n_expert + 1; ++i) {
        if (count_data[i] != expected_counts[(size_t) i]) {
            std::fprintf(stderr, "count mismatch i=%d got=%d expected=%d\n",
                    i, count_data[i], expected_counts[(size_t) i]);
            ggml_free(ctx);
            return 1;
        }
    }

    for (int e = 0; e < n_expert; ++e) {
        mask_data[e] = 1;
    }
    std::memset(counts->data, 0, ggml_nbytes(counts));
    reset_fetch_calls();
    if (!compute()) {
        ggml_free(ctx);
        return 1;
    }

    stock_data = static_cast<const float *>(stock->data);
    cold_data = static_cast<const float *>(cold->data);
    for (int64_t i = 0; i < ggml_nelements(stock); ++i) {
        if (!nearly_equal(cold_data[i], stock_data[i])) {
            std::fprintf(stderr, "all-cold mismatch i=%lld got=%g expected=%g\n",
                    (long long) i, cold_data[i], stock_data[i]);
            ggml_free(ctx);
            return 1;
        }
    }
    for (int e = 0; e < n_expert; ++e) {
        if (g_fetch_calls[e].load(std::memory_order_relaxed) != 1) {
            std::fprintf(stderr, "all-cold fetch-hook calls mismatch expert=%d got=%d expected=1\n",
                    e, g_fetch_calls[e].load(std::memory_order_relaxed));
            ggml_free(ctx);
            return 1;
        }
    }

    count_data = static_cast<const int32_t *>(counts->data);
    for (int i = 0; i < n_expert + 1; ++i) {
        if (count_data[i] != expected_counts[(size_t) i]) {
            std::fprintf(stderr, "second count mismatch i=%d got=%d expected=%d\n",
                    i, count_data[i], expected_counts[(size_t) i]);
            ggml_free(ctx);
            return 1;
        }
    }

    ggml_free(ctx);
    return 0;
}

int main() {
    ggml_set_moe_fetch_hook(count_fetch);
    const uint64_t cold_before = ggml_moe_cold_timer_us();
    const uint64_t mmid_before = ggml_moe_cold_mmid_timer_us();
    for (const int64_t n_tokens : {16, 17, 256}) {
        const int rc = run_case(n_tokens);
        if (rc != 0) {
            std::fprintf(stderr, "wackMall cold ops failed for %lld tokens\n", (long long) n_tokens);
            return rc;
        }
    }
    for (const int64_t n_tokens : {1, 16, 17}) {
        const int rc = run_fused_case(n_tokens);
        if (rc != 0) {
            std::fprintf(stderr, "wackMall fused cold op failed for %lld tokens\n", (long long) n_tokens);
            return rc;
        }
    }
    const uint64_t cold_after = ggml_moe_cold_timer_us();
    const uint64_t mmid_after = ggml_moe_cold_mmid_timer_us();
    if (mmid_after <= mmid_before || cold_after < cold_before ||
            cold_after - cold_before < mmid_after - mmid_before) {
        std::fprintf(stderr, "cold timer split invalid: total=%llu mmid=%llu\n",
                (unsigned long long) (cold_after - cold_before),
                (unsigned long long) (mmid_after - mmid_before));
        return 1;
    }
    ggml_set_moe_fetch_hook(nullptr);
    std::puts("wackMall cold ops: OK (MMID 16/17/256, fused 1/16/17 tokens)");
    return 0;
}
