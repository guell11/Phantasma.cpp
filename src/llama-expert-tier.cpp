#include "llama-expert-tier.h"
#include "llama-expert-tier-policy.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#else
#define NOMINMAX
#include <malloc.h>
#include <intrin.h>
#include <windows.h>
#include <BaseTsd.h> // SSIZE_T
typedef SSIZE_T ssize_t; // POSIX type not provided on Windows
#endif

// expert tiering hooks live in the CPU backend (ggml-cpu.c). Under
// GGML_BACKEND_DL the backend is a runtime-loaded .so, so libllama cannot
// link the setters directly; resolve them once from the backend registry
// (same pattern as llama-context.cpp for ggml_backend_cpu_set_threadpool).
typedef void (*moe_predict_hook_fn)(const ggml_tensor *, const ggml_tensor *);
typedef const char * (*moe_addr_hook_fn)(const void *, int64_t, const char *);

typedef void (*moe_set_predict_fn)(moe_predict_hook_fn);
typedef void (*moe_set_addr_fn)(moe_addr_hook_fn);
typedef void (*moe_set_route_fn)(FILE *, int);
typedef uint64_t (*moe_timer_fn)(void);

static moe_set_predict_fn g_fn_predict = NULL;
static moe_set_addr_fn    g_fn_probe   = NULL;
static moe_set_addr_fn    g_fn_fetch   = NULL;
static moe_set_route_fn   g_fn_route   = NULL;
static moe_timer_fn       g_fn_timer   = NULL;
static moe_timer_fn       g_fn_mmid_timer = NULL;

static void tier_resolve_moe_hooks(void) {
    if (g_fn_predict) {
        return;
    }
    ggml_backend_t backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backend_cpu) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (!reg) {
        return;
    }
    g_fn_predict = (moe_set_predict_fn) ggml_backend_reg_get_proc_address(reg, "ggml_set_moe_predict_hook");
    g_fn_probe   = (moe_set_addr_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_set_moe_probe_hook");
    g_fn_fetch   = (moe_set_addr_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_set_moe_fetch_hook");
    g_fn_route   = (moe_set_route_fn)   ggml_backend_reg_get_proc_address(reg, "ggml_set_route_trace");
    g_fn_timer   = (moe_timer_fn)       ggml_backend_reg_get_proc_address(reg, "ggml_moe_cold_timer_us");
    g_fn_mmid_timer = (moe_timer_fn)    ggml_backend_reg_get_proc_address(reg, "ggml_moe_cold_mmid_timer_us");
}

#define MOE_PREDICT_HOOK(fn)   do { if (g_fn_predict) g_fn_predict((fn)); } while (0)
#define MOE_PROBE_HOOK(fn)     do { if (g_fn_probe)   g_fn_probe((fn));   } while (0)
#define MOE_FETCH_HOOK(fn)     do { if (g_fn_fetch)   g_fn_fetch((fn));   } while (0)
#define MOE_ROUTE_HOOK(f, n)   do { if (g_fn_route)   g_fn_route((f), (n)); } while (0)
#define MOE_TIMER_HOOK()       (g_fn_timer ? g_fn_timer() : 0)
#define MOE_MMID_TIMER_HOOK()  (g_fn_mmid_timer ? g_fn_mmid_timer() : 0)

// portable atomic access to single i32s inside plain buffers (lut_host must
// stay a plain i32 vector: it is uploaded wholesale into the GPU lut tensor)
static inline int32_t tier_atomic_load_i32(const int32_t * p) {
#if defined(_MSC_VER)
    return _InterlockedExchangeAdd((volatile long *) p, 0);
#else
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#endif
}

static inline void tier_atomic_store_i32(int32_t * p, int32_t v) {
#if defined(_MSC_VER)
    _InterlockedExchange((volatile long *) p, (long) v);
#else
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
#endif
}

// tier status must print at default verbosity; libllama INFO requires -v
#define TIER_LOG(...) fprintf(stderr, __VA_ARGS__)

// page hints on mmap'd source weights: drop = free the file-backed pages of a
// GPU-resident expert (bytes refault from the model file on next touch).
// never call on malloc'd buffers - DONTNEED would zero anonymous pages.
static void tier_madvise(const void * p, size_t len, bool drop) {
#if !defined(_WIN32)
    static const long page = sysconf(_SC_PAGESIZE);
    const uintptr_t a = (uintptr_t) p & ~(uintptr_t) (page - 1);
    const uintptr_t b = ((uintptr_t) p + len + page - 1) & ~(uintptr_t) (page - 1);
    if (b > a) {
        madvise((void *) a, b - a, drop ? MADV_DONTNEED : MADV_WILLNEED);
    }
#else
    (void) p; (void) len; (void) drop;
#endif
}

// pread helper: portable persistent read for demand fetch
// POSIX: pread() on a shared read-only fd (thread-safe)
// Windows: ReadFile+OVERLAPPED (compile-guarded, marked UNTESTED)
static int g_pread_fd = -1;
#if defined(_WIN32)
static HANDLE g_pread_handle = INVALID_HANDLE_VALUE;
#endif
static bool g_pread_disabled = false; // KAT failure or mmap-off disables the path
static const char * g_pread_mmap_base = nullptr;
static size_t g_pread_mmap_size = 0;

static bool tier_pread_init(const char * path) {
    if (g_pread_disabled) {
        return false;
    }
#if !defined(_WIN32)
    g_pread_fd = open(path, O_RDONLY);
    if (g_pread_fd < 0) {
        TIER_LOG("%s: pread open failed: %s\n", __func__, strerror(errno));
        g_pread_disabled = true;
        return false;
    }
#else
    g_pread_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (g_pread_handle == INVALID_HANDLE_VALUE) {
        TIER_LOG("%s: pread CreateFileA failed\n", __func__);
        g_pread_disabled = true;
        return false;
    }
#endif
    return true;
}

static void tier_pread_close() {
#if !defined(_WIN32)
    if (g_pread_fd >= 0) {
        close(g_pread_fd);
        g_pread_fd = -1;
    }
#else
    if (g_pread_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(g_pread_handle);
        g_pread_handle = INVALID_HANDLE_VALUE;
    }
#endif
    g_pread_mmap_base = nullptr;
    g_pread_mmap_size = 0;
}

// pread a list of (offset, size) pairs into dest; returns total bytes read or -1 on error
// coalesces adjacent reads (offset + size == next offset) into a single call
static ssize_t tier_pread_list(void * dest, const std::vector<std::pair<size_t, size_t>> & spans) {
    if (g_pread_disabled || spans.empty()) {
        return -1;
    }
    char * dp = (char *) dest;
    ssize_t total = 0;
    size_t i = 0;
    while (i < spans.size()) {
        size_t off = spans[i].first;
        size_t sz = spans[i].second;
        size_t j = i + 1;
        while (j < spans.size() && spans[j].first == off + sz) {
            sz += spans[j].second;
            j++;
        }
#if !defined(_WIN32)
        ssize_t n = pread(g_pread_fd, dp, sz, off);
        if (n != (ssize_t) sz) {
            return -1;
        }
#else
        OVERLAPPED ov = {};
        ov.Offset = (DWORD) off;
        ov.OffsetHigh = (DWORD) (off >> 32);
        DWORD n = 0;
        if (!ReadFile(g_pread_handle, dp, (DWORD) sz, &n, &ov)) {
            return -1;
        }
        if (n != (DWORD) sz) {
            return -1;
        }
#endif
        dp += sz;
        total += (ssize_t) sz;
        i = j;
    }
    return total;
}

// KAT: pread the first 4 KiB of a host expert tensor, byte-compare vs the
// mmap. offset 0 is unusable: the loader munmaps the metadata fragment after
// load (llama_model_loader::load_all_data), so the header pages are no
// longer backed by the file mapping. the tensor span also validates the
// (data - mmap_base) offset math the fetch path relies on.
static bool tier_pread_kat(const void * tensor_data, size_t file_off) {
    if (g_pread_disabled || !tensor_data) {
        return false;
    }
    const size_t K = 4096;
    char buf[K];
    ssize_t n = tier_pread_list(buf, {{file_off, K}});
    if (n != (ssize_t) K) {
        TIER_LOG("%s: KAT pread failed\n", __func__);
        g_pread_disabled = true;
        return false;
    }
    if (memcmp(buf, tensor_data, K) != 0) {
        TIER_LOG("%s: KAT byte mismatch\n", __func__);
        g_pread_disabled = true;
        return false;
    }
    return true;
}

namespace llama_expert_tier {

std::vector<int> hot_slot_budget_policy(
        const std::vector<hot_slot_policy_layer> & layers,
        size_t byte_budget, int uniform_slots) {
    std::vector<int> result(layers.size(), 0);
    if (uniform_slots <= 0) {
        return result;
    }

    bool has_signal = false;
    size_t uniform_bytes = 0;
    for (const auto & L : layers) {
        if (L.expert_bytes == 0) {
            continue;
        }
        if (L.misses > L.reuses) {
            has_signal = true;
        }
        for (const float score : L.scores) {
            if (score > 0.0f) {
                has_signal = true;
                break;
            }
        }
        if (L.expert_bytes > SIZE_MAX/(size_t) uniform_slots ||
            uniform_bytes > SIZE_MAX - L.expert_bytes*(size_t) uniform_slots) {
            return result;
        }
        uniform_bytes += L.expert_bytes*(size_t) uniform_slots;
    }
    if (!has_signal) {
        if (uniform_bytes > byte_budget) {
            return result;
        }
        for (size_t il = 0; il < layers.size(); il++) {
            if (layers[il].expert_bytes != 0) {
                result[il] = std::min(uniform_slots, (int) layers[il].scores.size());
            }
        }
        return result;
    }

    std::vector<std::vector<float>> scores(layers.size());
    for (size_t il = 0; il < layers.size(); il++) {
        scores[il] = layers[il].scores;
        std::sort(scores[il].begin(), scores[il].end(), std::greater<float>());
    }

    size_t used = 0;
    for (;;) {
        int best_layer = -1;
        double best_value = -1.0;
        for (size_t il = 0; il < layers.size(); il++) {
            const auto & L = layers[il];
            const int slot = result[il];
            if (L.expert_bytes == 0 || slot >= (int) scores[il].size() ||
                L.expert_bytes > byte_budget - used) {
                continue;
            }
            const uint64_t residual = L.misses > L.reuses ? L.misses - L.reuses : 0;
            const double demand = std::max(0.0f, scores[il][slot]) +
                    (double) residual/(double) (slot + 1);
            const double value = demand/(double) L.expert_bytes;
            if (value > best_value || (value == best_value && (best_layer < 0 || (int) il < best_layer))) {
                best_value = value;
                best_layer = (int) il;
            }
        }
        if (best_layer < 0 || best_value <= 0.0) {
            break;
        }
        used += layers[best_layer].expert_bytes;
        result[best_layer]++;
    }
    return result;
}

static uint64_t g_h2d_dedupe_hits = 0;
static uint64_t g_h2d_slot_stalls = 0;
static uint64_t g_h2d_slot_stall_us = 0;
static uint64_t g_h2d_async_bytes = 0;
static uint64_t g_h2d_async_calls = 0;
static uint64_t g_h2d_sync_fallback_bytes = 0;
static uint64_t g_h2d_sync_fallback_calls = 0;
static uint64_t g_h2d_publish_us = 0;
static uint64_t g_h2d_publish_calls = 0;

struct h2d_stage {
    static constexpr size_t N_SLOTS = 3;

    struct slot {
        ggml_backend_buffer_t host = nullptr;
        ggml_backend_event_t  copy_done = nullptr;
        void *                data = nullptr;
        ggml_tensor *         dst = nullptr;
        size_t                offset = 0;
        size_t                size = 0;
        h2d_load_key          load_key = {};
        bool                  copy_in_flight = false;
        ggml_backend_t        wait_backend = nullptr;
    };

    ggml_backend_t        compute_backend = nullptr;
    ggml_backend_t        copy_backend = nullptr;
    slot                   slots[N_SLOTS];
    ggml_backend_event_t   compute_done = nullptr;
    size_t                capacity = 0;
    size_t                next_slot = 0;
    bool                  async    = false;
};

static ggml_backend_buffer_t tier_tensor_buffer(ggml_tensor * tensor) {
    if (!tensor) {
        return nullptr;
    }
    return tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
}

h2d_stage * h2d_stage_create(ggml_backend_t backend, size_t capacity) {
    if (!backend) {
        return nullptr;
    }

    h2d_stage * stage = new h2d_stage;
    stage->compute_backend = backend;
    stage->capacity = capacity;

    if (capacity == 0) {
        return stage;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        return stage;
    }

    ggml_backend_dev_props props = {};
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.async || !props.caps.events || !props.caps.host_buffer) {
        return stage;
    }

    stage->copy_backend = ggml_backend_dev_init(dev, nullptr);
    if (!stage->copy_backend || ggml_backend_get_device(stage->copy_backend) != dev) {
        if (stage->copy_backend) {
            ggml_backend_free(stage->copy_backend);
            stage->copy_backend = nullptr;
        }
        return stage;
    }

    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (!host_buft) {
        ggml_backend_free(stage->copy_backend);
        stage->copy_backend = nullptr;
        return stage;
    }

    for (size_t i = 0; i < h2d_stage::N_SLOTS; ++i) {
        h2d_stage::slot & slot = stage->slots[i];
        slot.host = ggml_backend_buft_alloc_buffer(host_buft, capacity);
        if (!slot.host || ggml_backend_buffer_get_type(slot.host) != host_buft) {
            break;
        }
        slot.data = ggml_backend_buffer_get_base(slot.host);
        slot.copy_done = ggml_backend_event_new(dev);
        if (!slot.data || !slot.copy_done) {
            break;
        }
        if (i + 1 == h2d_stage::N_SLOTS) {
            stage->compute_done = ggml_backend_event_new(dev);
        }
        if (i + 1 == h2d_stage::N_SLOTS && stage->compute_done) {
            stage->async = true;
        }
    }
    if (!stage->async) {
        for (h2d_stage::slot & slot : stage->slots) {
            if (slot.copy_done) {
                ggml_backend_event_free(slot.copy_done);
            }
            if (slot.host) {
                ggml_backend_buffer_free(slot.host);
            }
            slot = {};
        }
        if (stage->compute_done) {
            ggml_backend_event_free(stage->compute_done);
            stage->compute_done = nullptr;
        }
        ggml_backend_free(stage->copy_backend);
        stage->copy_backend = nullptr;
    }
    return stage;
}

static void h2d_stage_wait_slot(h2d_stage::slot & slot, bool count_stall) {
    if (!slot.copy_in_flight) {
        return;
    }
    const int64_t t0 = count_stall ? ggml_time_us() : 0;
    ggml_backend_event_synchronize(slot.copy_done);
    if (count_stall) {
        g_h2d_slot_stalls++;
        g_h2d_slot_stall_us += (uint64_t) (ggml_time_us() - t0);
    }
    slot.copy_in_flight = false;
    slot.dst = nullptr;
    slot.offset = 0;
    slot.size = 0;
    slot.load_key = {};
    slot.wait_backend = nullptr;
}

void h2d_stage_wait(h2d_stage * stage) {
    if (!stage) {
        return;
    }
    for (h2d_stage::slot & slot : stage->slots) {
        h2d_stage_wait_slot(slot, false);
    }
}

static void h2d_stage_fence_compute(h2d_stage * stage) {
    if (!stage->async) {
        return;
    }
    ggml_backend_event_record(stage->compute_done, stage->compute_backend);
    ggml_backend_event_synchronize(stage->compute_done);
}

void h2d_stage_free(h2d_stage * stage) {
    if (!stage) {
        return;
    }
    h2d_stage_wait(stage);
    h2d_stage_fence_compute(stage);
    for (h2d_stage::slot & slot : stage->slots) {
        if (slot.copy_done) {
            ggml_backend_event_free(slot.copy_done);
        }
        if (slot.host) {
            ggml_backend_buffer_free(slot.host);
        }
    }
    if (stage->compute_done) {
        ggml_backend_event_free(stage->compute_done);
    }
    if (stage->copy_backend) {
        ggml_backend_free(stage->copy_backend);
    }
    delete stage;
}

bool h2d_stage_is_async(const h2d_stage * stage) {
    return stage && stage->async;
}

size_t h2d_stage_capacity(const h2d_stage * stage) {
    return stage ? stage->capacity : 0;
}

bool h2d_stage_upload_once(h2d_stage * stage, ggml_tensor * dst, size_t offset, const void * src, size_t size, h2d_load_key key) {
    if (!stage || !dst || (!src && size != 0) || offset > ggml_nbytes(dst) || size > ggml_nbytes(dst) - offset) {
        return false;
    }
    if (size == 0) {
        return true;
    }

    ggml_backend_buffer_t dst_buffer = tier_tensor_buffer(dst);
    const bool compatible = dst_buffer &&
            stage->copy_backend &&
            ggml_backend_buffer_get_type(dst_buffer) == ggml_backend_get_default_buffer_type(stage->copy_backend) &&
            ggml_backend_get_default_buffer_type(stage->copy_backend) == ggml_backend_get_default_buffer_type(stage->compute_backend);
    if (!stage->async || !compatible || size > stage->capacity) {
        h2d_stage_wait(stage);
        if (stage->async) {
            h2d_stage_fence_compute(stage);
        } else {
            ggml_backend_synchronize(stage->compute_backend);
        }
        ggml_backend_tensor_set(dst, src, offset, size);
        g_h2d_sync_fallback_bytes += size;
        g_h2d_sync_fallback_calls++;
        return true;
    }

    if (key.identity != 0) {
        for (h2d_stage::slot & slot : stage->slots) {
            if (!slot.copy_in_flight || slot.load_key.identity != key.identity ||
                    slot.load_key.generation != key.generation) {
                continue;
            }
            if (slot.dst != dst || slot.offset != offset || slot.size != size) {
                return false;
            }
            g_h2d_dedupe_hits++;
            return true;
        }
    }

    h2d_stage::slot & slot = stage->slots[stage->next_slot];
    h2d_stage_wait_slot(slot, true);
    memcpy(slot.data, src, size);
    ggml_backend_tensor_set_async(stage->copy_backend, dst, slot.data, offset, size);
    g_h2d_async_bytes += size;
    g_h2d_async_calls++;
    ggml_backend_event_record(slot.copy_done, stage->copy_backend);
    ggml_backend_event_wait(stage->compute_backend, slot.copy_done);
    slot.dst = dst;
    slot.offset = offset;
    slot.size = size;
    slot.load_key = key;
    slot.copy_in_flight = true;
    slot.wait_backend = nullptr;
    stage->next_slot = (stage->next_slot + 1) % h2d_stage::N_SLOTS;
    return true;
}

bool h2d_stage_upload(h2d_stage * stage, ggml_tensor * dst, size_t offset, const void * src, size_t size) {
    return h2d_stage_upload_once(stage, dst, offset, src, size, {});
}

bool h2d_stage_wait_backend(h2d_stage * stage, ggml_backend_t backend) {
    if (!stage || !stage->async || !backend ||
            ggml_backend_get_device(backend) != ggml_backend_get_device(stage->compute_backend)) {
        return false;
    }
    bool waited = false;
    for (h2d_stage::slot & slot : stage->slots) {
        if (!slot.copy_in_flight) {
            continue;
        }
        if (backend != stage->compute_backend && slot.wait_backend != backend) {
            ggml_backend_event_wait(backend, slot.copy_done);
            slot.wait_backend = backend;
        }
        waited = true;
    }
    return waited;
}

struct store {
    ggml_tensor * w_hot  = nullptr; // [ne0, ne1, n_slots] on GPU, sentinel slot zeroed
    ggml_tensor * lut    = nullptr; // i32 [n_expert] on GPU: expert -> hot slot | sentinel
    ggml_tensor * mask   = nullptr; // i32 [n_expert] on CPU: 1 = cold
    ggml_tensor * counts = nullptr; // i32 [n_expert+1] on CPU: selection stats
    ggml_tensor * ptrs   = nullptr; // i64 [n_expert] on CPU: weight source address (pool slot | mmap)
    int  il      = -1;
    bool is_down = false;
    bool discardable = false; // mmap-backed and not mlocked: pages may be dropped
    bool poolable    = false; // weights_discardable regardless of LLAMA_EXPERT_MADVISE
    size_t pool_off = 0;      // offset of this tensor's slice within a pool slot
    std::vector<uint64_t> h2d_slot_identity;
};

// per-layer grouping for stats and online repin
struct layer_tier {
    int il       = -1;
    int n_expert = 0;
    int n_slots  = 0; // incl. sentinel slot
    int sentinel = 0;
    store * sd = nullptr; // store holding the down weight (counts source)
    std::vector<std::pair<const ggml_tensor *, store *>> ws;
    std::vector<int32_t> slot_expert; // [n_slots], -1 = empty
    std::vector<uint64_t> hot_generation; // [n_slots], bumped before every hot-slot rewrite
    std::vector<int32_t> lut_host;    // [n_expert]
    std::vector<float>   score;       // [n_expert], cumulative (LLAMA_EXPERT_DECAY)
    std::vector<float>   score_pp;    // [n_expert], PP-only EMA
    std::vector<float>   score_tg;    // [n_expert], TG-only EMA
    std::vector<uint64_t> cum;        // [n_expert], exact counts for persistence
    std::vector<uint64_t> cum_pp;     // [n_expert], exact PP counts
    std::vector<uint64_t> cum_tg;     // [n_expert], exact TG counts
    struct markov_edge {
        int32_t expert = -1;
        uint32_t count = 0;
    };
    std::vector<std::vector<markov_edge>> markov; // [from expert], sparse top-N rows
    std::vector<uint8_t> markov_prev;             // [expert], routed in previous window
    std::vector<uint8_t> markov_pending;          // [expert], predicted for this window
    int markov_pred_index = -1;
    std::vector<int32_t> dwell;       // [n_slots]
    uint64_t cum_cold = 0, cum_total = 0, cum_graphs = 0;
    // RAM pool: slot = one expert across all tiered tensors of this layer
    char * pool = nullptr;
    size_t pool_slot_bytes = 0;
    int    n_pool_slots = 0;
    std::vector<int32_t> pool_slot_expert; // [n_pool_slots], -1 = empty
    std::vector<int32_t> pool_dwell;       // [n_pool_slots]
    // residency contract: 0 free / 1 filling / 2 ready / 3 resident.
    // the update() window owns FREE/READY/RESIDENT transitions; the prefetch
    // worker only claims FREE->FILLING->READY mid-step (atomic CAS).
    std::unique_ptr<std::atomic<int32_t>[]> pool_slot_state; // [n_pool_slots]
    std::unique_ptr<std::atomic<int32_t>[]> pool_lut;        // [n_expert], -1 = not pooled
    // spec pool (prefetch): worker-only cache for predicted experts, probe-served
    char * poolB = nullptr;
    int    n_slotsB = 0;
    std::vector<int32_t> ownersB;                // [n_slotsB], -1 = empty
    std::unique_ptr<std::atomic<int32_t>[]> stateB; // [n_slotsB] 0 free/1 filling/2 ready/3 resident
    std::unique_ptr<std::atomic<int32_t>[]> lutB;   // [n_expert], -1 = not in B
    std::unique_ptr<std::atomic<int64_t>[]> lastB;  // [n_slotsB] last-predicted step
};

static int  g_S        = 16;
static int  g_tmax     = 16;
static bool g_adapt    = false;
static bool g_madvise  = true; // LLAMA_EXPERT_MADVISE=0 disables page hints
static bool g_score_split = false;
static float g_score_tg_weight = 0.5f;
static bool g_layer_budget_policy = false;
static bool g_markov_predict = false;
static int  g_markov_top_n = 2;
static thread_local bool g_hot_only = false; // graph-build state set by begin_moe_cold per layer

struct hot_moe_state {
    const ggml_tensor * gate_w = nullptr;
    const ggml_tensor * up_w   = nullptr;
    const ggml_tensor * down_w = nullptr;
    const store * gate = nullptr;
    const store * up   = nullptr;
    const store * down = nullptr;
};

static thread_local hot_moe_state g_hot_moe;

static ggml_context * g_ctx_gpu = nullptr; // owned for process lifetime
static ggml_context * g_ctx_cpu = nullptr;
static std::atomic<const llama_model *> g_owner_model{nullptr};
static std::mutex g_init_mu;
static std::atomic<bool> g_tier_disabled{false};
static bool g_tier_conflict_logged = false;
static thread_local const llama_model * g_graph_model = nullptr;

graph_scope::graph_scope(const llama_model & model) : previous(g_graph_model) {
    g_graph_model = &model;
}

graph_scope::~graph_scope() {
    g_graph_model = previous;
}

static bool tier_graph_active() {
    return !g_tier_disabled.load(std::memory_order_acquire) &&
        g_graph_model == g_owner_model.load(std::memory_order_acquire);
}

static std::unordered_map<const ggml_tensor *, store> g_stores;
static std::vector<layer_tier> g_layers;
static uint64_t g_repins = 0; // hot-set changes since init
static uint64_t g_h2d_next_identity = 1;

static h2d_stage *     g_h2d_stage = nullptr;
static ggml_backend_t  g_h2d_compute_backend = nullptr;

static size_t   g_pool_bytes = 0; // LLAMA_EXPERT_RAMPOOL budget in bytes (0 = off)
static size_t   g_pool_alloc = 0; // resident pool bytes
static size_t   g_pool_fill_budget = 0; // per-update() fill budget
static uint64_t g_pool_fills = 0, g_pool_evictions = 0;
static uint64_t g_pool_hits = 0, g_pool_cold = 0;

static uint64_t g_fetch_us = 0;   // cumulative wall time in pool_fill() memcpy
static uint64_t g_steps = 0;      // update() calls (= graph computes)
static FILE *   g_route_log = nullptr; // actual-routing trace (co-opened with pred log)
static bool     g_stats_enabled = false;

struct dispatcher_cost_ema {
    double value = 0.0;
    uint64_t samples = 0;
};

struct perf_mode_stats {
    uint64_t steps = 0;
    uint64_t tokens = 0;
    uint64_t total_us = 0;
    uint64_t wait_us = 0;
    uint64_t update_us = 0;
    uint64_t cold_us = 0;
    uint64_t cold_mmid_us = 0;
    uint64_t h2d_us = 0;
    uint64_t h2d_bytes = 0;
    uint64_t h2d_calls = 0;
    uint64_t h2d_slot_stalls = 0;
    uint64_t h2d_slot_stall_us = 0;
    uint64_t h2d_async_bytes = 0;
    uint64_t h2d_async_calls = 0;
    uint64_t h2d_sync_fallback_bytes = 0;
    uint64_t h2d_sync_fallback_calls = 0;
    uint64_t h2d_publish_us = 0;
    uint64_t h2d_publish_calls = 0;
    uint64_t route_total = 0;
    uint64_t route_cold = 0;
    dispatcher_cost_ema dispatcher_hot_gpu_route_us;
    dispatcher_cost_ema dispatcher_cold_cpu_mmid_route_us;
    dispatcher_cost_ema dispatcher_graph_publish_wait_us;
    dispatcher_cost_ema dispatcher_h2d_us_per_mib;
};

static perf_mode_stats g_perf_pp;
static perf_mode_stats g_perf_tg;
static uint64_t g_perf_last_cold_us = 0;
static uint64_t g_perf_last_cold_mmid_us = 0;
static uint64_t g_perf_last_h2d_us = 0;
static uint64_t g_perf_last_h2d_bytes = 0;
static uint64_t g_perf_last_h2d_calls = 0;
static uint64_t g_perf_last_h2d_slot_stalls = 0;
static uint64_t g_perf_last_h2d_slot_stall_us = 0;
static uint64_t g_perf_last_h2d_async_bytes = 0;
static uint64_t g_perf_last_h2d_async_calls = 0;
static uint64_t g_perf_last_h2d_sync_fallback_bytes = 0;
static uint64_t g_perf_last_h2d_sync_fallback_calls = 0;
static uint64_t g_perf_last_h2d_publish_us = 0;
static uint64_t g_perf_last_h2d_publish_calls = 0;
static uint64_t g_perf_pending_route_total = 0;
static uint64_t g_perf_pending_route_cold = 0;
static uint64_t g_h2d_us = 0;
static uint64_t g_h2d_bytes = 0;
static uint64_t g_h2d_calls = 0;
static uint64_t g_h2d_lut_us = 0;
static uint64_t g_h2d_lut_bytes = 0;
static uint64_t g_h2d_lut_calls = 0;

static void tier_h2d_stage_shutdown() {
    h2d_stage_free(g_h2d_stage);
    g_h2d_stage = nullptr;
    if (g_h2d_compute_backend) {
        ggml_backend_free(g_h2d_compute_backend);
        g_h2d_compute_backend = nullptr;
    }
}

static perf_mode_stats & perf_mode(int64_t n_tokens) {
    return n_tokens == 1 ? g_perf_tg : g_perf_pp;
}

static void dispatcher_cost_update(dispatcher_cost_ema & cost, double sample) {
    static constexpr double alpha = 0.125;
    if (cost.samples == 0) {
        cost.value = sample;
    } else {
        cost.value += alpha*(sample - cost.value);
    }
    cost.samples++;
}

static bool dispatcher_cost_ready(const perf_mode_stats & ps) {
    static constexpr uint64_t min_samples = 8;
    return ps.dispatcher_hot_gpu_route_us.samples >= min_samples &&
        ps.dispatcher_cold_cpu_mmid_route_us.samples >= min_samples &&
        ps.dispatcher_graph_publish_wait_us.samples >= min_samples &&
        ps.dispatcher_h2d_us_per_mib.samples >= min_samples;
}

static void tier_tensor_set_profiled(
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size,
        bool is_lut) {
    if (!g_stats_enabled) {
        ggml_backend_tensor_set(tensor, data, offset, size);
        return;
    }

    const int64_t t0 = ggml_time_us();
    ggml_backend_tensor_set(tensor, data, offset, size);
    const uint64_t dt = (uint64_t) (ggml_time_us() - t0);
    if (is_lut) {
        g_h2d_lut_us += dt;
        g_h2d_lut_bytes += size;
        g_h2d_lut_calls++;
    } else {
        g_h2d_us += dt;
        g_h2d_bytes += size;
        g_h2d_calls++;
    }
}

static bool tier_tensor_set_promotion(
        ggml_tensor * tensor,
        const void * data,
        size_t offset,
        size_t size,
        h2d_load_key key) {
    if (!g_h2d_stage) {
        tier_tensor_set_profiled(tensor, data, offset, size, false);
        return false;
    }

    const int64_t t0 = g_stats_enabled ? ggml_time_us() : 0;
    const uint64_t dedupe_before = g_stats_enabled ? g_h2d_dedupe_hits : 0;
    const bool ok = key.identity != 0 ?
        h2d_stage_upload_once(g_h2d_stage, tensor, offset, data, size, key) :
        h2d_stage_upload(g_h2d_stage, tensor, offset, data, size);
    const bool deduped = g_stats_enabled && key.identity != 0 && g_h2d_dedupe_hits != dedupe_before;
    if (g_stats_enabled && ok && !deduped) {
        g_h2d_us += (uint64_t) (ggml_time_us() - t0);
        g_h2d_bytes += size;
        g_h2d_calls++;
    }
    if (!ok) {
        tier_tensor_set_profiled(tensor, data, offset, size, false);
    }
    return ok;
}

static void tier_h2d_stage_publish(ggml_backend_t backend) {
    if (!g_h2d_stage) {
        return;
    }
    const int64_t t0 = g_stats_enabled ? ggml_time_us() : 0;
    if (!h2d_stage_wait_backend(g_h2d_stage, backend)) {
        h2d_stage_wait(g_h2d_stage);
    }
    if (g_stats_enabled) {
        const uint64_t dt = (uint64_t) (ggml_time_us() - t0);
        g_h2d_us += dt;
        g_h2d_publish_us += dt;
        g_h2d_publish_calls++;
    }
}

// pre-gate predictor: immutable CPU mirrors of each MoE layer's router and
// ffn norm weights. The fused cold op reports its router input x through the
// predict hook; running the NEXT MoE layer's router on x (with its norm
// exactly reconstructed as w_next * (x / w_cur)) predicts that layer's
// experts one layer ahead of demand. No learned parameters: the predictor
// is the model's own gating function applied early.
struct pred_layer {
    int il = -1;
    int n_routed = 0;
    std::vector<float> norm;
    std::vector<float> gate;
};

static std::vector<pred_layer> g_pred;
static std::unordered_map<const void *, int> g_pred_ix; // counts->data -> g_pred index
static bool g_predict = false; // LLAMA_EXPERT_PREDICT=1 enables
static FILE * g_pred_log = nullptr; // LLAMA_EXPERT_PREDICT_LOG (debug)
static uint64_t g_pred_pushes = 0;

// prefetch probe: lets a cold op read an expert straight from a READY
// (fully memcpy'd, not yet published) pool slot instead of the mmap source.
// keyed by the store's ptrs->data; content is byte-identical either way, so
// a probe miss costs speed, never correctness.
struct probe_ctx {
    const std::atomic<int32_t> * lut;
    const std::atomic<int32_t> * states;
    const int32_t * owners;
    const char * pool;
    int64_t slot_bytes;
    int64_t pool_off;
};
static std::unordered_map<const void *, probe_ctx> g_probe_ix; // ptrs->data -> ctx
static std::atomic<uint64_t> g_probe_hits{0}, g_probe_miss{0};

static bool g_dirty = false;
static std::string g_sidecar_path;
static uint64_t g_fingerprint = 0;

static const char * pregate_probe(const void * key, int64_t e, const char * fallback) {
    auto it = g_probe_ix.find(key);
    if (it == g_probe_ix.end()) {
        return fallback;
    }
    const probe_ctx & c = it->second;
    const int32_t k = c.lut[e].load(std::memory_order_acquire);
    if (k >= 0 && c.states[k].load(std::memory_order_acquire) >= 2 && c.owners[k] == (int32_t) e) {
        g_probe_hits.fetch_add(1, std::memory_order_relaxed);
        return c.pool + (size_t) k*c.slot_bytes + c.pool_off;
    }
    g_probe_miss.fetch_add(1, std::memory_order_relaxed);
    return fallback;
}

static int prefetch_resident_slot(const layer_tier & L, int e) {
    if (!L.poolB || e < 0 || e >= L.n_expert) {
        return -1;
    }
    const int32_t k = L.lutB[e].load(std::memory_order_acquire);
    if (k < 0 || k >= L.n_slotsB ||
            L.stateB[k].load(std::memory_order_acquire) != 3 ||
            L.ownersB[k] != e) {
        return -1;
    }
    return k;
}

static void prefetch_release(layer_tier & L, int e, int k) {
    if (k < 0 || k >= L.n_slotsB || L.ownersB[k] != e ||
            L.stateB[k].load(std::memory_order_acquire) != 3) {
        return;
    }
    int32_t expected = k;
    if (!L.lutB[e].compare_exchange_strong(expected, -1, std::memory_order_acq_rel)) {
        return;
    }
    L.ownersB[k] = -1;
    L.stateB[k].store(0, std::memory_order_release);
}

static uint64_t fnv1a_64(const void * data, size_t len, uint64_t h = 14695981039346656037ULL) {
    const uint8_t * p = (const uint8_t *) data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static uint64_t fnv1a_64_str(const char * s, uint64_t h = 14695981039346656037ULL) {
    return fnv1a_64(s, strlen(s), h);
}

static uint64_t fnv1a_64_u64(uint64_t v, uint64_t h = 14695981039346656037ULL) {
    return fnv1a_64(&v, sizeof(v), h);
}

static uint64_t compute_fingerprint(const llama_model & model) {
    uint64_t h = 14695981039346656037ULL;
    const char * arch = llm_arch_name(model.arch);
    h = fnv1a_64_str(arch ? arch : "unknown", h);
    h = fnv1a_64_u64((uint64_t) model.size(), h);
    h = fnv1a_64_u64((uint64_t) model.n_tensors(), h);
    for (const auto & kv : model.tensors_by_name) {
        h = fnv1a_64_str(kv.first.c_str(), h);
        const ggml_tensor * t = kv.second;
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            h = fnv1a_64_u64((uint64_t) t->ne[i], h);
        }
    }
    return h;
}

static void save_sidecar();

static std::vector<std::vector<float>> load_sidecar(const llama_model & model) {
    std::vector<std::vector<float>> result;
    if (g_sidecar_path.empty()) {
        return result;
    }
    std::ifstream in(g_sidecar_path, std::ios::binary);
    if (!in) {
        TIER_LOG("%s: no sidecar at %s\n", __func__, g_sidecar_path.c_str());
        return result;
    }
    uint32_t version = 0;
    uint64_t fp = 0;
    in.read((char *) &version, sizeof(version));
    in.read((char *) &fp, sizeof(fp));
    if (!in || version != 1) {
        TIER_LOG("%s: sidecar version mismatch (got %u, want 1), ignoring\n", __func__, version);
        return result;
    }
    if (fp != g_fingerprint) {
        TIER_LOG("%s: sidecar fingerprint mismatch (got %016llx, want %016llx), ignoring\n",
                __func__, (unsigned long long) fp, (unsigned long long) g_fingerprint);
        return result;
    }
    const int n_layer = model.hparams.n_layer();
    const int n_expert = model.hparams.n_expert;
    result.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        result[il].resize(n_expert);
        in.read((char *) result[il].data(), n_expert * sizeof(float));
        if (!in) {
            TIER_LOG("%s: sidecar truncated at layer %d, ignoring\n", __func__, il);
            result.clear();
            return result;
        }
    }
    TIER_LOG("%s: loaded sidecar %s (fingerprint %016llx)\n", __func__, g_sidecar_path.c_str(), (unsigned long long) fp);
    return result;
}

static void save_sidecar() {
    if (!g_dirty || g_sidecar_path.empty() || g_fingerprint == 0) {
        return;
    }
    size_t n_expert = 0;
    for (const auto & L : g_layers) {
        if (!L.score.empty()) {
            n_expert = L.score.size();
            break;
        }
    }
    if (n_expert == 0) {
        return;
    }
    const std::string tmp = g_sidecar_path + ".tmp";
    std::ofstream out(tmp, std::ios::binary);
    if (!out) {
        TIER_LOG("%s: cannot create %s\n", __func__, tmp.c_str());
        return;
    }
    const uint32_t version = 1;
    out.write((const char *) &version, sizeof(version));
    out.write((const char *) &g_fingerprint, sizeof(g_fingerprint));
    const std::vector<float> zero_score(n_expert, 0.0f);
    for (const auto & L : g_layers) {
        const std::vector<float> & score = L.score.empty() ? zero_score : L.score;
        if (score.size() != n_expert) {
            out.close();
            std::remove(tmp.c_str());
            return;
        }
        out.write((const char *) score.data(), n_expert * sizeof(float));
    }
    out.close();
    if (!out) {
        TIER_LOG("%s: write failed for %s\n", __func__, tmp.c_str());
        std::remove(tmp.c_str());
        return;
    }
#if defined(_WIN32)
    const bool ok = MoveFileExA(tmp.c_str(), g_sidecar_path.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    const bool ok = std::rename(tmp.c_str(), g_sidecar_path.c_str()) == 0;
#endif
    if (!ok) {
        TIER_LOG("%s: rename failed for %s\n", __func__, g_sidecar_path.c_str());
        std::remove(tmp.c_str());
    } else {
        TIER_LOG("%s: saved sidecar %s (fingerprint %016llx)\n", __func__, g_sidecar_path.c_str(), (unsigned long long) g_fingerprint);
    }
}

// demand-fetch staging ring (LLAMA_EXPERT_PREAD): bounded, page-aligned,
// owned by the cold path, separate from the score-managed demand pool. the
// first compute thread to touch a pool-miss expert claims a slot and preads
// the expert's slices (all tiered tensors of its layer) from the model file;
// threads claiming different experts pread in parallel; a thread needing an
// in-flight expert waits on the slot word. a slot is reclaimable only by a
// claim from a different layer's op: cold ops are serialized by the
// threadpool, so a slot keyed by another layer has no live readers. no
// pool/lut/score metadata is touched here; update() stays the single writer.
// slot word: bits [0:2) state (0 free / 1 claimed / 2 ready), [2:34) expert,
// [34:64) layer - one atomic word so claim+key is a single CAS.
static char * g_stage_base = nullptr;
static size_t g_stage_stride = 0; // page-aligned slot stride
static int    g_stage_n = 0;
static std::unique_ptr<std::atomic<uint64_t>[]> g_stage_word; // [g_stage_n]
static const char * g_stage_mmap = nullptr; // model mmap base (offset math)
static std::atomic<uint64_t> g_stage_fetches{0}, g_stage_bytes{0};
static std::atomic<uint64_t> g_stage_stalls{0}, g_stage_fallbacks{0}, g_stage_fails{0};
static bool g_stage_fail_test = false; // LLAMA_EXPERT_PREAD_FAIL (test-only)

struct stage_ctx {
    int          il;
    int64_t      pool_off; // this tensor's offset within a slot
    const char * w_data;   // mmap base of the tensor (residency check)
    int64_t      slice;    // per-expert bytes of this tensor
};
static std::unordered_map<const void *, stage_ctx> g_stage_ix; // ptrs->data -> ctx

static uint64_t stage_word(int il, int64_t e, uint64_t st) {
    return ((uint64_t) il << 34) | ((uint64_t) e << 2) | st;
}

static const char * stage_fetch(const void * key, int64_t e, const char * fallback) {
    auto it = g_stage_ix.find(key);
    if (it == g_stage_ix.end()) {
        return fallback;
    }
    const stage_ctx & c = it->second;
    if (fallback != c.w_data + e*c.slice) {
        return fallback; // resident: demand pool or a READY spec slot
    }
    const uint64_t mykey = stage_word(c.il, e, 0);
    for (int attempt = 0; attempt < 4; attempt++) {
        int cand = -1;
        uint64_t candw = 0;
        for (int k = 0; k < g_stage_n; k++) {
            const uint64_t w = g_stage_word[k].load(std::memory_order_acquire);
            const uint64_t st = w & 3;
            if (st != 0 && (w & ~3ull) == mykey) {
                if (st == 2) {
                    return g_stage_base + (size_t) k*g_stage_stride + c.pool_off;
                }
                // claimed by another thread: wait for the fill to finish
                g_stage_stalls.fetch_add(1, std::memory_order_relaxed);
                uint64_t v = w;
                int spins = 0;
                while ((v = g_stage_word[k].load(std::memory_order_acquire)) == w) {
                    if ((++spins & 0x3FF) == 0) {
                        std::this_thread::yield();
                    }
                }
                if ((v & ~3ull) == mykey && (v & 3) == 2) {
                    return g_stage_base + (size_t) k*g_stage_stride + c.pool_off;
                }
                g_stage_fallbacks.fetch_add(1, std::memory_order_relaxed);
                return fallback; // fill failed: mmap fallback
            }
            if (cand < 0 && (st == 0 || (st == 2 && (int) (w >> 34) != c.il))) {
                cand  = k;
                candw = w;
            }
        }
        if (cand < 0) {
            break; // ring full of live slots: mmap fallback
        }
        if (!g_stage_word[cand].compare_exchange_strong(candw, mykey | 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            continue; // lost the claim race: rescan
        }
        char * slot = g_stage_base + (size_t) cand*g_stage_stride;
        size_t total = 0;
        bool ok = !g_stage_fail_test;
        if (ok) {
            const layer_tier & L = g_layers[c.il];
            for (const auto & kv : L.ws) {
                const size_t slice = ggml_nbytes(kv.first)/L.n_expert;
                const size_t off = (size_t) ((const char *) kv.first->data - g_stage_mmap) + (size_t) e*slice;
                if (tier_pread_list(slot + kv.second->pool_off, {{off, slice}}) != (ssize_t) slice) {
                    ok = false;
                    break;
                }
                total += slice;
            }
        }
        if (ok) {
            g_stage_word[cand].store(mykey | 2, std::memory_order_release);
            g_stage_fetches.fetch_add(1, std::memory_order_relaxed);
            g_stage_bytes.fetch_add(total, std::memory_order_relaxed);
            return slot + c.pool_off;
        }
        g_stage_word[cand].store(0, std::memory_order_release);
        const uint64_t nf = g_stage_fails.fetch_add(1, std::memory_order_relaxed) + 1;
        if (nf <= 3 || (nf & (nf - 1)) == 0) { // rate-limited
            TIER_LOG("%s: pread failed (%llu total), mmap fallback\n", __func__, (unsigned long long) nf);
        }
        return fallback;
    }
    g_stage_fallbacks.fetch_add(1, std::memory_order_relaxed);
    return fallback;
}

// prefetch worker: drains predicted (layer, expert) pairs mid-step, claiming
// FREE pool slots via CAS (FILLING), memcpy'ing from the mmap source, then
// release-storing READY. owns state 1 only; the update() window owns 0/2/3.
static std::deque<std::pair<int32_t, int32_t>> g_work_q; // (g_pred target index, expert)
static std::mutex g_work_mu;
static std::condition_variable g_work_cv;
static bool g_work_exit = false;
static std::vector<std::thread> g_workers;
static std::atomic<int64_t> g_work_inflight{0};
static int64_t g_work_budget = 64*1024*1024; // LLAMA_EXPERT_PREFETCH_MB
static size_t g_poolB_bytes = 0;   // LLAMA_EXPERT_PREFETCH_GB (default 0 = off)
static size_t g_poolB_alloc = 0;
static int g_n_workers = 2;        // LLAMA_EXPERT_PREFETCH_THREADS (clamp 1..8)
static std::atomic<uint64_t> g_pred_step{0}; // window tick for lastB timestamps
static uint64_t g_predB_evict = 0; // timestamp evictions by the window
static std::atomic<uint64_t> g_pred_fills{0};
static uint64_t g_pred_published = 0;
static std::atomic<uint64_t> g_pred_pread_reads{0}, g_pred_pread_bytes{0}, g_pred_pread_fallbacks{0};
static std::atomic<uint64_t> g_pred_drop_hot{0}, g_pred_drop_dup{0}, g_pred_drop_budget{0}, g_pred_drop_full{0};
static uint64_t g_markov_transitions = 0, g_markov_predictions = 0;
static uint64_t g_markov_actual = 0, g_markov_covered = 0;
static uint64_t g_markov_useful = 0, g_markov_late = 0, g_markov_waste = 0;
static uint64_t g_markov_drop_hot = 0, g_markov_drop_resident = 0;
static uint64_t g_markov_drop_dup = 0, g_markov_drop_full = 0;

static void prefetch_fill(int pi, int e) {
    layer_tier & L = g_layers[g_pred[pi].il];
    if (!L.poolB) {
        return;
    }
    if (tier_atomic_load_i32(&L.lut_host[e]) != L.sentinel) {
        g_pred_drop_hot++;
        return; // hot already
    }
    if (L.pool_lut[e].load(std::memory_order_acquire) >= 0) {
        g_pred_drop_dup++;
        return; // in the demand pool already
    }
    const int64_t step = (int64_t) g_pred_step.load(std::memory_order_relaxed);
    const int32_t k0 = L.lutB[e].load(std::memory_order_acquire);
    if (k0 >= 0) {
        L.lastB[k0].store(step, std::memory_order_release); // refresh on re-prediction
        g_pred_drop_dup++;
        return;
    }
    if (g_work_inflight.load(std::memory_order_relaxed) + (int64_t) L.pool_slot_bytes > g_work_budget) {
        g_pred_drop_budget++;
        return;
    }
    // claim a FREE slot; never evict - the window keeps ~25% of slots free
    int k = -1;
    for (int i = 0; i < L.n_slotsB; i++) {
        int32_t free_state = 0;
        if (L.stateB[i].compare_exchange_strong(free_state, 1, std::memory_order_acquire)) {
            k = i;
            break;
        }
    }
    if (k < 0) {
        g_pred_drop_full++;
        return;
    }
    int32_t expect = -1;
    if (!L.lutB[e].compare_exchange_strong(expect, k, std::memory_order_release)) {
        // another worker won; free the slot, refresh the winner's timestamp
        L.stateB[k].store(0, std::memory_order_release);
        L.lastB[expect].store(step, std::memory_order_release);
        g_pred_drop_dup++;
        return;
    }
    L.ownersB[k] = e;
    L.lastB[k].store(step, std::memory_order_relaxed);
    g_work_inflight.fetch_add((int64_t) L.pool_slot_bytes, std::memory_order_relaxed);
    for (auto & kv : L.ws) {
        store & st = *kv.second;
        const size_t slice = ggml_nbytes(kv.first)/L.n_expert;
        char * dst = L.poolB + (size_t) k*L.pool_slot_bytes + st.pool_off;
        const char * src = (const char *) kv.first->data + slice*e;
        bool loaded = false;
        if (g_pread_mmap_base) {
            const uintptr_t base = (uintptr_t) g_pread_mmap_base;
            const uintptr_t addr = (uintptr_t) src;
            const size_t off = addr >= base ? (size_t) (addr - base) : SIZE_MAX;
            if (off <= g_pread_mmap_size && slice <= g_pread_mmap_size - off) {
                loaded = tier_pread_list(dst, {{off, slice}}) == (ssize_t) slice;
                if (loaded) {
                    g_pred_pread_reads.fetch_add(1, std::memory_order_relaxed);
                    g_pred_pread_bytes.fetch_add(slice, std::memory_order_relaxed);
                }
            }
        }
        if (!loaded) {
            memcpy(dst, src, slice);
            g_pred_pread_fallbacks.fetch_add(1, std::memory_order_relaxed);
        }
    }
    L.stateB[k].store(2, std::memory_order_release);
    g_work_inflight.fetch_sub((int64_t) L.pool_slot_bytes, std::memory_order_relaxed);
    g_pred_fills++;
}

static void prefetch_worker() {
    std::unique_lock<std::mutex> lk(g_work_mu);
    for (;;) {
        g_work_cv.wait(lk, [] { return g_work_exit || !g_work_q.empty(); });
        if (g_work_exit) {
            return;
        }
        const auto req = g_work_q.front();
        g_work_q.pop_front();
        lk.unlock();
        prefetch_fill(req.first, req.second);
        lk.lock();
    }
}

static void pred_shutdown() {
    if (!g_workers.empty()) {
        { std::lock_guard<std::mutex> lk(g_work_mu); g_work_exit = true; }
        g_work_cv.notify_all();
        for (auto & t : g_workers) {
            t.join();
        }
    }
}

// called from ggml_compute_forward_moe_cold at ith==0
static void pregate_hook(const ggml_tensor * counts, const ggml_tensor * x) {
    static uint64_t seq = 0;
    const uint64_t cur_seq = seq++; // unconditional: one per op call, aligned with the trace seq
    if (!counts || !counts->data || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float)) {
        return;
    }
    auto it = g_pred_ix.find(counts->data);
    if (it == g_pred_ix.end() || it->second + 1 >= (int) g_pred.size()) {
        return;
    }
    const pred_layer & cur = g_pred[it->second];
    const pred_layer & nxt = g_pred[it->second + 1];
    const int64_t ne0 = x->ne[0];
    if ((int64_t) cur.norm.size() != ne0 || (int64_t) nxt.norm.size() != ne0) {
        return;
    }
    const int64_t n_exp = (int64_t) (nxt.gate.size()/ne0);
    thread_local std::vector<float> hv, lg;
    thread_local std::vector<int32_t> idx;
    thread_local std::vector<uint8_t> queued;
    hv.resize(ne0);
    lg.resize(n_exp);
    idx.resize(n_exp);
    queued.assign(n_exp, 0);
    const int K = nxt.n_routed;
    const int64_t n_tokens = x->ne[2];
    int queued_jobs = 0;
    for (int64_t t = 0; t < n_tokens; t++) {
        const float * xt = (const float *) ((const char *) x->data + t*x->nb[2]);
        for (int64_t i = 0; i < ne0; i++) {
            const float w = cur.norm[i];
            hv[i] = nxt.norm[i]*(w*w > 1e-12f ? xt[i]/w : 0.0f);
        }
        for (int64_t e = 0; e < n_exp; e++) {
            const float * g = nxt.gate.data() + (size_t) e*ne0;
            float s = 0.0f;
            for (int64_t i = 0; i < ne0; i++) {
                s += g[i]*hv[i];
            }
            lg[e] = s;
        }
        for (int64_t e = 0; e < n_exp; e++) {
            idx[e] = (int32_t) e;
        }
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
        if (g_pred_log) {
            fprintf(g_pred_log, "%llu,%lld,%d", (unsigned long long) cur_seq, (long long) n_tokens, nxt.il);
            for (int k = 0; k < K; k++) {
                fprintf(g_pred_log, ",%d", idx[k]);
            }
            fprintf(g_pred_log, "\n");
        }
        if (!g_workers.empty()) {
            std::lock_guard<std::mutex> lk(g_work_mu);
            if (g_work_q.size() < 4096) {
                layer_tier & LT = g_layers[nxt.il];
                for (int k = 0; k < K; k++) {
                    const int32_t e = idx[k];
                    if (queued[e]) {
                        continue;
                    }
                    queued[e] = 1;
                    if (LT.pool_lut && LT.pool_lut[e].load(std::memory_order_relaxed) >= 0) {
                        continue; // already in demand pool
                    }
                    g_work_q.emplace_back((int32_t) (it->second + 1), e);
                    queued_jobs++;
                }
            }
        }
        g_pred_pushes++;
    }
    if (queued_jobs == 1) {
        g_work_cv.notify_one();
    } else if (queued_jobs > 1) {
        g_work_cv.notify_all();
    }
}

static void pred_init(const llama_model & model) {
    if (!g_predict) {
        if (g_markov_predict) {
            TIER_LOG("%s: Markov prefetch disabled (LLAMA_EXPERT_PREDICT=1 required)\n", __func__);
        }
        return;
    }
    const int n_layer = model.hparams.n_layer();
    std::vector<float> raw;
    size_t mirror_bytes = 0;
    for (int il = 0; il < n_layer; il++) {
        if (g_layers[il].ws.empty() || !g_layers[il].sd) {
            continue;
        }
        const ggml_tensor * gate = model.layers[il].ffn_gate_inp;
        // pre-router norm: ffn_norm on classic archs, attn_post_norm on the
        // hybrid qwen35/qwen36 archs (GGUF post_attention_norm)
        const ggml_tensor * norm = model.layers[il].ffn_norm;
        if (!norm) {
            norm = model.layers[il].attn_post_norm;
        }
        if (!gate || !norm || gate->type != GGML_TYPE_F32 || norm->type != GGML_TYPE_F32) {
            continue;
        }
        const int64_t ne0 = gate->ne[0];
        const int64_t n_exp = gate->ne[1];
        if (n_exp != g_layers[il].n_expert || norm->ne[0] != ne0) {
            continue;
        }
        pred_layer pl;
        pl.il = il;
        pl.n_routed = model.hparams.n_expert_used;
        pl.norm.resize(ne0);
        pl.gate.resize((size_t) n_exp*ne0);
        ggml_backend_tensor_get(norm, pl.norm.data(), 0, ne0*sizeof(float));
        ggml_backend_tensor_get(gate, pl.gate.data(), 0, (size_t) n_exp*ne0*sizeof(float));
        const int pred_index = (int) g_pred.size();
        g_pred_ix[g_layers[il].sd->counts->data] = pred_index;
        g_layers[il].markov_pred_index = pred_index;
        g_pred.push_back(std::move(pl));
        mirror_bytes += (size_t) (n_exp + 1)*ne0*sizeof(float);
    }
    const bool want_worker = g_predict && g_poolB_bytes > 0 && g_pred.size() >= 2;
    if (g_pred.size() >= 2 && (g_pred_log || want_worker)) {
        tier_resolve_moe_hooks();
        MOE_PREDICT_HOOK(pregate_hook);
    }
    if (want_worker) {
        for (auto & kv : g_stores) {
            store & st = kv.second;
            if (!st.ptrs || !st.ptrs->data) {
                continue;
            }
            layer_tier & L = g_layers[st.il];
            if (!L.poolB || L.n_slotsB <= 0) {
                continue;
            }
            probe_ctx c{ L.lutB.get(), L.stateB.get(), L.ownersB.data(),
                         L.poolB, (int64_t) L.pool_slot_bytes, (int64_t) st.pool_off };
            g_probe_ix[st.ptrs->data] = c;
        }
        tier_resolve_moe_hooks();
        MOE_PROBE_HOOK(pregate_probe);
        for (int i = 0; i < g_n_workers; i++) {
            g_workers.emplace_back(prefetch_worker);
        }
        atexit(pred_shutdown); // registered after dump_stats: runs first (reverse order)
    }
    char wrk[64] = "";
    if (want_worker) {
        snprintf(wrk, sizeof(wrk), ", prefetching (%d threads)", g_n_workers);
    }
    TIER_LOG("%s: pre-gate predictor: %zu layers, %.1f MiB mirrors%s\n", __func__,
            g_pred.size(), (double) mirror_bytes/(1024.0*1024.0),
            want_worker ? wrk : (g_pred.size() >= 2 && g_pred_log ? ", logging" : " (inactive)"));
    if (g_markov_predict && !want_worker) {
        TIER_LOG("%s: Markov prefetch disabled (needs eligible layers and LLAMA_EXPERT_PREFETCH_GB)\n", __func__);
    }
}



// RAM pool (LLAMA_EXPERT_RAMPOOL): per-layer contiguous blocks, one tier
// below VRAM. cold ops read weights through store::ptrs (pool slot or mmap
// fallback). all mutations happen on the host between compute steps (post
// sched sync); the compute threads only read the tables, lock-free.
#define POOL_ALIGN 64

static size_t pool_align(size_t n) {
    return (n + POOL_ALIGN - 1) & ~(size_t) (POOL_ALIGN - 1);
}

// point every store's ptrs[e] at the expert's current location
static void pool_publish(layer_tier & L, int e) {
    const int k = L.pool_lut[e];
    for (auto & kv : L.ws) {
        store & st = *kv.second;
        if (!st.ptrs) {
            continue;
        }
        const size_t slice = ggml_nbytes(kv.first)/L.n_expert;
        const char * addr = k >= 0 ? L.pool + (size_t) k*L.pool_slot_bytes + st.pool_off
                                   : (const char *) kv.first->data + slice*e;
        ((int64_t *) st.ptrs->data)[e] = (int64_t) (uintptr_t) addr;
    }
}

// copy expert e into pool slot k and publish it; consume an already-resident
// prefetch slot when available, otherwise fall back to the mmap source
static void pool_fill(layer_tier & L, int k, int e) {
    int32_t free_state = 0;
    if (!L.pool_slot_state[k].compare_exchange_strong(free_state, 3, std::memory_order_acquire)) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    const int prefetched = prefetch_resident_slot(L, e);
    for (auto & kv : L.ws) {
        store & st = *kv.second;
        const size_t slice = ggml_nbytes(kv.first)/L.n_expert;
        const char * mmap_src = (const char *) kv.first->data + slice*e;
        const char * src = prefetched >= 0 ?
                L.poolB + (size_t) prefetched*L.pool_slot_bytes + st.pool_off :
                mmap_src;
        memcpy(L.pool + (size_t) k*L.pool_slot_bytes + st.pool_off, src, slice);
        if (st.poolable) {
            tier_madvise(mmap_src, slice, true);
        }
    }
    prefetch_release(L, e, prefetched);
    g_fetch_us += (uint64_t)(ggml_time_us() - t0);
    L.pool_slot_expert[k] = e;
    L.pool_lut[e] = k;
    L.pool_dwell[k] = 0;
    pool_publish(L, e);
    g_pool_fills++;
}

static void pool_evict(layer_tier & L, int k) {
    const int e = L.pool_slot_expert[k];
    if (e < 0) {
        return;
    }
    L.pool_slot_expert[k] = -1;
    L.pool_lut[e] = -1;
    L.pool_slot_state[k].store(0, std::memory_order_release);
    L.pool_dwell[k] = 0;
    pool_publish(L, e); // back to the mmap fallback
    g_pool_evictions++;
}

static bool parse_heat_csv(const std::string & path, int n_layer,
        std::vector<std::vector<std::pair<int64_t, int32_t>>> & heat) {
    std::ifstream in(path);
    if (!in) {
        LLAMA_LOG_ERROR("%s: cannot open heat csv '%s'\n", __func__, path.c_str());
        return false;
    }
    heat.resize(n_layer);
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string tok;
        std::vector<int64_t> vals;
        while (std::getline(ss, tok, ',')) {
            vals.push_back(std::stoll(tok));
        }
        if (vals.size() < 3) {
            continue;
        }
        const int il = (int) vals[0];
        if (il < 0 || il >= n_layer) {
            continue;
        }
        heat[il].emplace_back(vals[2], (int32_t) vals[1]);
    }
    return true;
}

float score_ema_update(float score, float decay, uint64_t count) {
    return score*decay + (float) count;
}

static float placement_score(const layer_tier & L, int e) {
    if (!g_score_split) {
        return L.score[e];
    }
    return (1.0f - g_score_tg_weight)*L.score_pp[e] + g_score_tg_weight*L.score_tg[e];
}

static void markov_add_edge(std::vector<layer_tier::markov_edge> & row, int e) {
    for (auto & edge : row) {
        if (edge.expert == e) {
            if (edge.count != UINT32_MAX) {
                edge.count++;
            }
            return;
        }
    }
    if ((int) row.size() < g_markov_top_n) {
        row.push_back({ e, 1 });
        return;
    }
    auto victim = std::min_element(row.begin(), row.end(), [](const auto & a, const auto & b) {
        return a.count != b.count ? a.count < b.count : a.expert > b.expert;
    });
    victim->expert = e;
    victim->count = victim->count == UINT32_MAX ? UINT32_MAX : victim->count + 1;
}

static void markov_update(layer_tier & L, const int32_t * cnt, int n) {
    if (!g_markov_predict || g_workers.empty() || !L.poolB || !L.lutB || !L.stateB || L.markov_pred_index < 0 ||
        (int) L.markov.size() != n || (int) L.markov_prev.size() != n ||
        (int) L.markov_pending.size() != n) {
        return;
    }

    for (int e = 0; e < n; e++) {
        const uint64_t routed = (uint64_t) std::max(0, cnt[e]);
        g_markov_actual += routed;
        if (!L.markov_pending[e]) {
            continue;
        }
        L.markov_pending[e] = 0;
        if (routed == 0) {
            g_markov_waste++;
            continue;
        }
        g_markov_covered += routed;
        const int32_t k = L.lutB[e].load(std::memory_order_acquire);
        if (k >= 0 && L.ownersB[k] == e && L.stateB[k].load(std::memory_order_acquire) >= 2) {
            g_markov_useful++;
        } else {
            g_markov_late++;
        }
    }

    thread_local std::vector<int32_t> active;
    active.clear();
    if ((int) active.capacity() < n) {
        active.reserve(n);
    }
    for (int e = 0; e < n; e++) {
        if (cnt[e] > 0) {
            active.push_back(e);
        }
    }
    for (int from = 0; from < n; from++) {
        if (!L.markov_prev[from]) {
            continue;
        }
        for (const int to : active) {
            markov_add_edge(L.markov[from], to);
            g_markov_transitions++;
        }
    }
    std::fill(L.markov_prev.begin(), L.markov_prev.end(), 0);
    for (const int e : active) {
        L.markov_prev[e] = 1;
    }

    struct candidate {
        int32_t expert;
        uint64_t count;
    };
    thread_local std::vector<candidate> candidates;
    candidates.clear();
    if ((int) candidates.capacity() < n) {
        candidates.reserve(n);
    }
    for (const int from : active) {
        for (const auto & edge : L.markov[from]) {
            auto it = std::find_if(candidates.begin(), candidates.end(), [&](const candidate & c) {
                return c.expert == edge.expert;
            });
            if (it == candidates.end()) {
                candidates.push_back({ edge.expert, edge.count });
            } else {
                it->count += edge.count;
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        return a.count != b.count ? a.count > b.count : a.expert < b.expert;
    });

    int queued_jobs = 0;
    {
        std::lock_guard<std::mutex> lk(g_work_mu);
        for (int i = 0; i < (int) candidates.size() && i < g_markov_top_n; i++) {
            const int e = candidates[i].expert;
            if (tier_atomic_load_i32(&L.lut_host[e]) != L.sentinel) {
                g_markov_drop_hot++;
                continue;
            }
            if (L.pool_lut[e].load(std::memory_order_acquire) >= 0) {
                g_markov_drop_resident++;
                continue;
            }
            if (L.markov_pending[e] || L.lutB[e].load(std::memory_order_acquire) >= 0) {
                g_markov_drop_dup++;
                continue;
            }
            if (g_work_q.size() >= 4096) {
                g_markov_drop_full++;
                break;
            }
            L.markov_pending[e] = 1;
            g_work_q.emplace_back(L.markov_pred_index, e);
            g_markov_predictions++;
            queued_jobs++;
        }
    }
    if (queued_jobs == 1) {
        g_work_cv.notify_one();
    } else if (queued_jobs > 1) {
        g_work_cv.notify_all();
    }
}

// consume selection counts, update decayed scores + cumulative stats, and
// (LLAMA_EXPERT_ADAPT) repin at most one slot. called at graph build time,
// i.e. between compute steps; counts are written by the fused cold op.
static void maybe_update(layer_tier & L, ggml_backend_t promotion_backend, int64_t n_tokens) {
    if (!L.sd || !L.sd->counts->data) {
        return;
    }
    int32_t * cnt = (int32_t *) L.sd->counts->data;
    const int n = L.n_expert;
    if (cnt[n] == 0) {
        return; // no compute since last visit
    }
    const int32_t * mask = (const int32_t *) L.sd->mask->data;

    markov_update(L, cnt, n);

    static const float decay = [] {
        const char * e = getenv("LLAMA_EXPERT_DECAY");
        return e ? (float) atof(e) : 0.999f;
    }();

    const bool is_tg = n_tokens == 1;
    for (int e = 0; e < n; e++) {
        if (cnt[e] > 0) {
            g_dirty = true;
        }
        L.score[e] = score_ema_update(L.score[e], decay, (uint64_t) cnt[e]);
        L.score_pp[e] = score_ema_update(L.score_pp[e], decay, is_tg ? 0 : (uint64_t) cnt[e]);
        L.score_tg[e] = score_ema_update(L.score_tg[e], decay, is_tg ? (uint64_t) cnt[e] : 0);
        L.cum[e]  += (uint64_t) cnt[e];
        if (is_tg) {
            L.cum_tg[e] += (uint64_t) cnt[e];
        } else {
            L.cum_pp[e] += (uint64_t) cnt[e];
        }
        if (mask[e]) {
            L.cum_cold += (uint64_t) cnt[e];
            if (L.pool) {
                g_pool_cold += (uint64_t) cnt[e];
                if (L.pool_lut[e] >= 0) {
                    g_pool_hits += (uint64_t) cnt[e];
                }
            }
        }
        cnt[e] = 0;
    }
    L.cum_total += (uint64_t) cnt[n];
    L.cum_graphs++;
    cnt[n] = 0;

    // publish completed prefetch fills (READY -> RESIDENT); recycle any whose
    // expert went hot mid-fill or that pool A picked up. then keep ~25% of
    // slots free for the worker by evicting the oldest-timestamped.
    if (L.poolB && L.n_slotsB > 0) {
        for (int k = 0; k < L.n_slotsB; k++) {
            if (L.stateB[k].load(std::memory_order_acquire) != 2) {
                continue;
            }
            const int32_t e = L.ownersB[k];
            if (tier_atomic_load_i32(&L.lut_host[e]) != L.sentinel || L.pool_lut[e].load(std::memory_order_relaxed) >= 0) {
                L.lutB[e].store(-1, std::memory_order_release);
                L.ownersB[k] = -1;
                L.stateB[k].store(0, std::memory_order_release);
                continue;
            }
            L.stateB[k].store(3, std::memory_order_release);
            g_pred_published++;
        }
        int free_n = 0;
        for (int k = 0; k < L.n_slotsB; k++) {
            if (L.stateB[k].load(std::memory_order_acquire) == 0) {
                free_n++;
            }
        }
        const int target = L.n_slotsB/4 + 1;
        while (free_n < target) {
            int kv = -1;
            int64_t tmin = INT64_MAX;
            for (int k = 0; k < L.n_slotsB; k++) {
                const int32_t st = L.stateB[k].load(std::memory_order_acquire);
                if (st != 2 && st != 3) {
                    continue;
                }
                const int64_t v = L.lastB[k].load(std::memory_order_acquire);
                if (v < tmin) {
                    tmin = v;
                    kv = k;
                }
            }
            if (kv < 0) {
                break;
            }
            const int32_t e = L.ownersB[kv];
            if (e >= 0) {
                L.lutB[e].store(-1, std::memory_order_release);
            }
            L.ownersB[kv] = -1;
            L.stateB[kv].store(0, std::memory_order_release);
            g_predB_evict++;
            free_n++;
        }
    }

    if (!g_adapt) {
        return;
    }

    for (int s = 0; s < L.n_slots; s++) {
        L.dwell[s]++;
    }

    // coldest hot slot (empty slots count as score 0)
    int si = -1;
    float smin = FLT_MAX;
    for (int s = 0; s < L.n_slots; s++) {
        if (s == L.sentinel) {
            continue;
        }
        const float v = L.slot_expert[s] < 0 ? 0.0f : placement_score(L, L.slot_expert[s]);
        if (v < smin) {
            smin = v;
            si = s;
        }
    }
    // hottest cold expert
    int ec = -1;
    float sc = 0.0f;
    for (int e = 0; e < n; e++) {
        if (mask[e] && placement_score(L, e) > sc) {
            sc = placement_score(L, e);
            ec = e;
        }
    }
    if (si >= 0 && ec >= 0) {
        const int eold = L.slot_expert[si];
        if (eold < 0 || (L.dwell[si] >= 32 && sc > 1.5f*smin)) {
            uint64_t generation = ++L.hot_generation[si];
            if (generation == 0) {
                generation = ++L.hot_generation[si];
            }
            const bool ec_pooled = L.pool && L.pool_lut[ec] >= 0;
            const int ec_prefetched = ec_pooled ? -1 : prefetch_resident_slot(L, ec);
            tier_atomic_store_i32(&L.lut_host[ec], si); // pairs with worker reads
            if (eold >= 0) {
                tier_atomic_store_i32(&L.lut_host[eold], L.sentinel);
            }
            for (auto & kv : L.ws) {
                const ggml_tensor * w = kv.first;
                store & st = *kv.second;
                const size_t slice = ggml_nbytes(w)/n;
                // H2D source: the pool copy when resident (its mmap pages are dropped)
                const char * srcp = ec_prefetched >= 0 ?
                        L.poolB + (size_t) ec_prefetched*L.pool_slot_bytes + st.pool_off :
                        st.ptrs ? (const char *) (uintptr_t) ((const int64_t *) st.ptrs->data)[ec]
                                : (const char *) w->data + slice*ec;
                const h2d_load_key key = {
                    st.h2d_slot_identity[(size_t) si],
                    generation,
                };
                tier_tensor_set_promotion(st.w_hot, srcp, slice*si, slice, key);
            }
            tier_h2d_stage_publish(promotion_backend);
            prefetch_release(L, ec, ec_prefetched);
            for (auto & kv : L.ws) {
                const ggml_tensor * w = kv.first;
                store & st = *kv.second;
                const size_t slice = ggml_nbytes(w)/n;
                if (st.discardable) {
                    tier_madvise((const char *) w->data + slice*ec, slice, true);
                    if (eold >= 0) {
                        tier_madvise((const char *) w->data + slice*eold, slice, false);
                    }
                }
                tier_tensor_set_profiled(st.lut, L.lut_host.data(), 0, n*sizeof(int32_t), true);
                int32_t * m = (int32_t *) st.mask->data;
                m[ec] = 0;
                if (eold >= 0) {
                    m[eold] = 1;
                }
            }
            if (ec_pooled) {
                // promotion consumed the pool copy; the slot goes back to empty.
                // a FILLING slot is the worker's: the publish pass recycles it
                // next window (the H2D above read ptrs = mmap, correct bytes).
                const int k = L.pool_lut[ec];
                if (L.pool_slot_state[k].load(std::memory_order_acquire) != 1) {
                    L.pool_slot_expert[k] = -1;
                    L.pool_slot_state[k].store(0, std::memory_order_release);
                    L.pool_lut[ec] = -1;
                    L.pool_dwell[k] = 0;
                    pool_publish(L, ec);
                    g_pool_evictions++;
                }
            }
            L.slot_expert[si] = ec;
            L.dwell[si] = 0;
            g_repins++;
        }
    }

    // RAM pool: same hysteresis shape as the hot tier, one level down
    if (L.pool && L.n_pool_slots > 0) {
        for (int k = 0; k < L.n_pool_slots; k++) {
            L.pool_dwell[k]++;
        }
        // coldest pooled slot (empty slots count as score 0); FILLING/READY
        // slots are the publish pass's (defensive: pool A never sees them)
        int ki = -1;
        float pmin = FLT_MAX;
        for (int k = 0; k < L.n_pool_slots; k++) {
            if (L.pool_slot_state[k].load(std::memory_order_acquire) == 1 ||
                L.pool_slot_state[k].load(std::memory_order_acquire) == 2) {
                continue;
            }
            const float v = L.pool_slot_expert[k] < 0 ? 0.0f : placement_score(L, L.pool_slot_expert[k]);
            if (v < pmin) {
                pmin = v;
                ki = k;
            }
        }
        // hottest cold expert that is neither hot nor pooled
        int ep = -1;
        float sp = 0.0f;
        for (int e = 0; e < n; e++) {
            if (mask[e] && L.pool_lut[e] < 0 && placement_score(L, e) > sp) {
                sp = placement_score(L, e);
                ep = e;
            }
        }
        if (ki >= 0 && ep >= 0 && g_pool_fill_budget >= L.pool_slot_bytes) {
            const int ev = L.pool_slot_expert[ki];
            if (ev < 0 || (L.pool_dwell[ki] >= 32 && sp > 1.5f*pmin)) {
                pool_evict(L, ki);
                pool_fill(L, ki, ep);
                g_pool_fill_budget -= L.pool_slot_bytes;
            }
        }
    }
}

static void dump_stats() {
    if (const char * p = getenv("LLAMA_EXPERT_STATS")) {
        FILE * f = strcmp(p, "1") ? fopen(p, "w") : stderr;
        if (f) {
            fprintf(f, "expert_tier: repins %llu\n", (unsigned long long) g_repins);
            if (g_pool_bytes) {
                fprintf(f, "expert_pool: fills %llu evictions %llu hits %llu / %llu cold (%.1f%%) resident %.2f GiB\n",
                        (unsigned long long) g_pool_fills, (unsigned long long) g_pool_evictions,
                        (unsigned long long) g_pool_hits, (unsigned long long) g_pool_cold,
                        g_pool_cold ? 100.0*(double) g_pool_hits/(double) g_pool_cold : 0.0,
                        (double) g_pool_alloc/(double) (1 << 30));
            }
            if (g_pred.size() >= 2) {
                fprintf(f, "expert_predict: pushes %llu fills %llu published %llu specevict %llu probe %llu/%llu drop hot %llu dup %llu budget %llu full %llu\n",
                        (unsigned long long) g_pred_pushes, (unsigned long long) g_pred_fills,
                        (unsigned long long) g_pred_published, (unsigned long long) g_predB_evict,
                        (unsigned long long) g_probe_hits.load(), (unsigned long long) g_probe_miss.load(),
                        (unsigned long long) g_pred_drop_hot, (unsigned long long) g_pred_drop_dup,
                        (unsigned long long) g_pred_drop_budget, (unsigned long long) g_pred_drop_full);
                fprintf(f, "expert_predict_pread: reads %llu bytes %llu fallbacks %llu\n",
                        (unsigned long long) g_pred_pread_reads.load(),
                        (unsigned long long) g_pred_pread_bytes.load(),
                        (unsigned long long) g_pred_pread_fallbacks.load());
            }
            if (g_markov_predict) {
                fprintf(f, "expert_markov: top_n %d transitions %llu predictions %llu coverage %llu/%llu (%.1f%%) useful %llu late %llu waste %llu drop hot %llu resident %llu dup %llu full %llu\n",
                        g_markov_top_n, (unsigned long long) g_markov_transitions,
                        (unsigned long long) g_markov_predictions,
                        (unsigned long long) g_markov_covered, (unsigned long long) g_markov_actual,
                        g_markov_actual ? 100.0*(double) g_markov_covered/(double) g_markov_actual : 0.0,
                        (unsigned long long) g_markov_useful, (unsigned long long) g_markov_late,
                        (unsigned long long) g_markov_waste, (unsigned long long) g_markov_drop_hot,
                        (unsigned long long) g_markov_drop_resident, (unsigned long long) g_markov_drop_dup,
                        (unsigned long long) g_markov_drop_full);
            }
            if (g_stage_n > 0) {
                fprintf(f, "expert_pread: fetches %llu bytes %llu stalls %llu fallbacks %llu fails %llu\n",
                        (unsigned long long) g_stage_fetches.load(),
                        (unsigned long long) g_stage_bytes.load(),
                        (unsigned long long) g_stage_stalls.load(),
                        (unsigned long long) g_stage_fallbacks.load(),
                        (unsigned long long) g_stage_fails.load());
            }
            {
                const uint64_t cold_us = MOE_TIMER_HOOK();
                fprintf(f, "expert_timers: steps %llu fetch %llu us (%.2f ms/step) cold_compute %llu us (%.2f ms/step)\n",
                        (unsigned long long) g_steps,
                        (unsigned long long) g_fetch_us,
                        g_steps ? (double) g_fetch_us / (double) g_steps / 1000.0 : 0.0,
                        (unsigned long long) cold_us,
                        g_steps ? (double) cold_us / (double) g_steps / 1000.0 : 0.0);
            }
            {
                const auto print_mode = [f](const char * mode, const perf_mode_stats & ps) {
                    if (ps.steps == 0 || ps.tokens == 0) {
                        return;
                    }
                    const double cache_hit_pct = ps.route_total ?
                        100.0 * (double) (ps.route_total - ps.route_cold) / (double) ps.route_total : 0.0;
                    const double h2d_gbps = ps.h2d_us ?
                        (double) ps.h2d_bytes / ((double) ps.h2d_us * 1000.0) : 0.0;
                    const double tok_s = ps.total_us ?
                        (double) ps.tokens * 1000000.0 / (double) ps.total_us : 0.0;

                    fprintf(f, "=== WACKMALL PERF ===\n");
                    fprintf(f, "mode=%s tokens=%llu steps=%llu\n", mode,
                            (unsigned long long) ps.tokens, (unsigned long long) ps.steps);
                    fprintf(f, "router_ms=NA\n");
                    fprintf(f, "moe_compute_ms=%.3f moe_compute_scope=cold_cpu_node_wall\n", ps.cold_us / 1000.0);
                    fprintf(f, "graph_wait_ms=%.3f\n", ps.wait_us / 1000.0);
                    fprintf(f, "tier_update_ms=%.3f\n", ps.update_us / 1000.0);
                    fprintf(f, "h2d_ms=%.3f\n", ps.h2d_us / 1000.0);
                    fprintf(f, "h2d_wait_ms=%.3f h2d_wait_scope=synchronous_promotion\n", ps.h2d_us / 1000.0);
                    fprintf(f, "h2d_bytes=%llu h2d_calls=%llu h2d_gbps=%.3f\n",
                            (unsigned long long) ps.h2d_bytes,
                            (unsigned long long) ps.h2d_calls,
                            h2d_gbps);
                    fprintf(f, "h2d_async_bytes=%llu h2d_async_calls=%llu\n",
                            (unsigned long long) ps.h2d_async_bytes,
                            (unsigned long long) ps.h2d_async_calls);
                    fprintf(f, "h2d_sync_fallback_bytes=%llu h2d_sync_fallback_calls=%llu\n",
                            (unsigned long long) ps.h2d_sync_fallback_bytes,
                            (unsigned long long) ps.h2d_sync_fallback_calls);
                    fprintf(f, "h2d_slot_stalls=%llu h2d_slot_stall_ms=%.3f h2d_slot_stall_scope=ring_reuse\n",
                            (unsigned long long) ps.h2d_slot_stalls,
                            ps.h2d_slot_stall_us / 1000.0);
                    fprintf(f, "h2d_publish_calls=%llu h2d_publish_wait_ms=%.3f h2d_publish_wait_scope=promotion_publish\n",
                            (unsigned long long) ps.h2d_publish_calls,
                            ps.h2d_publish_us / 1000.0);
                    fprintf(f, "dispatcher_policy=current_static dispatcher_cost_ready=%d dispatcher_min_samples=8\n",
                            dispatcher_cost_ready(ps) ? 1 : 0);
                    fprintf(f, "dispatcher_hot_gpu_route_us_ema=%.6f samples=%llu scope=graph_wait_per_hot_route_proxy\n",
                            ps.dispatcher_hot_gpu_route_us.value,
                            (unsigned long long) ps.dispatcher_hot_gpu_route_us.samples);
                    fprintf(f, "dispatcher_cold_cpu_mmid_route_us_ema=%.6f samples=%llu cold_mmid_ms=%.3f scope=compact_mmid_wall_per_cold_route\n",
                            ps.dispatcher_cold_cpu_mmid_route_us.value,
                            (unsigned long long) ps.dispatcher_cold_cpu_mmid_route_us.samples,
                            ps.cold_mmid_us / 1000.0);
                    fprintf(f, "dispatcher_graph_publish_wait_us_ema=%.6f samples=%llu scope=graph_sync_plus_promotion_publish_per_step\n",
                            ps.dispatcher_graph_publish_wait_us.value,
                            (unsigned long long) ps.dispatcher_graph_publish_wait_us.samples);
                    fprintf(f, "dispatcher_h2d_us_per_mib_ema=%.6f samples=%llu scope=h2d_submit_excluding_publish\n",
                            ps.dispatcher_h2d_us_per_mib.value,
                            (unsigned long long) ps.dispatcher_h2d_us_per_mib.samples);
                    fprintf(f, "cache_hit_pct=%.2f cache_scope=hot_vram_routes\n", cache_hit_pct);
                    fprintf(f, "duplicate_loads=%llu duplicate_loads_scope=h2d_inflight_explicit_key\n",
                            (unsigned long long) g_h2d_dedupe_hits);
                    if (strcmp(mode, "PP") == 0) {
                        fprintf(f, "pp_tok_s=%.3f\n", tok_s);
                    } else {
                        fprintf(f, "expert_wait_us_per_token=NA\n");
                        fprintf(f, "cold_compute_us_per_token=%.3f\n",
                                ps.tokens ? (double) ps.cold_us / (double) ps.tokens : 0.0);
                        fprintf(f, "tg_tok_s=%.3f\n", tok_s);
                    }
                };
                print_mode("PP", g_perf_pp);
                print_mode("TG", g_perf_tg);
                {
                    const uint64_t probe_hits = g_probe_hits.load();
                    const uint64_t probe_miss = g_probe_miss.load();
                    const double prefetch_pct = (probe_hits + probe_miss) ?
                        100.0 * (double) probe_hits / (double) (probe_hits + probe_miss) : 0.0;
                    fprintf(f, "prefetch_accuracy_pct=%.2f prefetch_accuracy_scope=weight_probe_global\n", prefetch_pct);
                }
                if (g_h2d_lut_calls) {
                    fprintf(f, "expert_h2d_lut: calls %llu bytes %llu time %.3f ms\n",
                            (unsigned long long) g_h2d_lut_calls,
                            (unsigned long long) g_h2d_lut_bytes,
                            g_h2d_lut_us / 1000.0);
                }
            }
            for (const auto & L : g_layers) {
                if (L.ws.empty() || L.cum_total == 0) {
                    continue;
                }
                uint64_t routes_pp = 0;
                uint64_t routes_tg = 0;
                for (int e = 0; e < L.n_expert; e++) {
                    routes_pp += L.cum_pp[e];
                    routes_tg += L.cum_tg[e];
                }
                fprintf(f, "layer %2d: cold %llu / %llu (%.1f%%) graphs %llu routes_pp %llu routes_tg %llu score_mode=%s\n", L.il,
                        (unsigned long long) L.cum_cold, (unsigned long long) L.cum_total,
                        100.0*(double) L.cum_cold/(double) L.cum_total,
                        (unsigned long long) L.cum_graphs,
                        (unsigned long long) routes_pp,
                        (unsigned long long) routes_tg,
                        g_score_split ? "split" : "aggregate");
            }
            // invariant: mask agrees with lut, slot<->expert bijection holds
            for (const auto & L : g_layers) {
                if (!L.sd) {
                    continue;
                }
                const int32_t * m = (const int32_t *) L.sd->mask->data;
                int bad = 0;
                for (int e = 0; e < L.n_expert; e++) {
                    const int want = (L.lut_host[e] == L.sentinel) ? 1 : 0;
                    if (m[e] != want) {
                        bad++;
                    }
                }
                for (int s = 0; s < L.n_slots; s++) {
                    if (s == L.sentinel || L.slot_expert[s] < 0) {
                        continue;
                    }
                    if (L.lut_host[L.slot_expert[s]] != s) {
                        bad++;
                    }
                }
                if (L.pool) {
                    for (int k = 0; k < L.n_pool_slots; k++) {
                        if (L.pool_slot_expert[k] >= 0 && L.pool_lut[L.pool_slot_expert[k]] != k) {
                            bad++;
                        }
                    }
                    for (int e = 0; e < L.n_expert; e++) {
                        const int k = L.pool_lut[e];
                        if (k >= 0 && L.pool_slot_expert[k] != e) {
                            bad++; // pool<->expert bijection, all claimed states
                        } else if (k >= 0 && L.pool_slot_state[k] == 3 && m[e] != 1) {
                            bad++; // resident implies cold
                        }
                    }
                }
                if (L.poolB) {
                    for (int e = 0; e < L.n_expert; e++) {
                        const int k = L.lutB[e];
                        if (k >= 0 && L.ownersB[k] != e) {
                            bad++;
                        }
                    }
                    for (int k = 0; k < L.n_slotsB; k++) {
                        if (L.ownersB[k] >= 0 && (int32_t) L.lutB[L.ownersB[k]].load() != k) {
                            bad++;
                        }
                    }
                }
                if (bad) {
                    fprintf(f, "layer %2d: INVARIANT VIOLATIONS %d\n", L.il, bad);
                }
            }
            if (f != stderr) {
                fclose(f);
            }
        }
    }
    if (const char * p = getenv("LLAMA_EXPERT_USAGE")) {
        FILE * f = fopen(p, "w");
        if (f) {
            fprintf(f, "layer,expert,count\n");
            for (const auto & L : g_layers) {
                if (L.cum.empty()) {
                    continue;
                }
                for (int e = 0; e < L.n_expert; e++) {
                    if (L.cum[e] > 0) {
                        fprintf(f, "%d,%d,%llu\n", L.il, e, (unsigned long long) L.cum[e]);
                    }
                }
            }
            fclose(f);
        }
    }
}

bool update(const llama_model & model, ggml_backend_t promotion_backend, int64_t n_tokens) {
    if (g_tier_disabled.load(std::memory_order_acquire) ||
            g_owner_model.load(std::memory_order_acquire) != &model) {
        return false;
    }

    bool has_pending = false;
    for (const auto & L : g_layers) {
        if (!L.sd || !L.sd->counts->data) {
            continue;
        }
        const int32_t * cnt = (const int32_t *) L.sd->counts->data;
        if (cnt[L.n_expert] != 0) {
            has_pending = true;
            break;
        }
    }
    if (!has_pending) {
        return false;
    }

    if (g_stats_enabled) {
        g_perf_pending_route_total = 0;
        g_perf_pending_route_cold = 0;
        for (const auto & L : g_layers) {
            if (!L.sd || !L.sd->counts->data || !L.sd->mask->data) {
                continue;
            }
            const int32_t * cnt = (const int32_t *) L.sd->counts->data;
            const int32_t * mask = (const int32_t *) L.sd->mask->data;
            for (int e = 0; e < L.n_expert; e++) {
                if (cnt[e] <= 0) {
                    continue;
                }
                g_perf_pending_route_total += (uint64_t) cnt[e];
                if (mask[e]) {
                    g_perf_pending_route_cold += (uint64_t) cnt[e];
                }
            }
        }
    }

    g_pred_step.fetch_add(1, std::memory_order_relaxed);
    g_steps++;
    for (auto & L : g_layers) {
        g_pool_fill_budget = 16 << 20;
        maybe_update(L, promotion_backend, n_tokens);
    }
    return true;
}

bool stats_enabled() {
    return g_stats_enabled;
}

void record_perf(int64_t n_tokens, uint64_t total_us, uint64_t wait_us, uint64_t update_us) {
    if (!g_stats_enabled || n_tokens <= 0) {
        return;
    }
    perf_mode_stats & ps = perf_mode(n_tokens);
    ps.steps++;
    ps.tokens += (uint64_t) n_tokens;
    ps.total_us += total_us;
    ps.wait_us += wait_us;
    ps.update_us += update_us;
    const uint64_t route_total = g_perf_pending_route_total;
    const uint64_t route_cold = g_perf_pending_route_cold;
    const uint64_t route_hot = route_total >= route_cold ? route_total - route_cold : 0;
    ps.route_total += route_total;
    ps.route_cold += route_cold;
    g_perf_pending_route_total = 0;
    g_perf_pending_route_cold = 0;

    const uint64_t cold_now = MOE_TIMER_HOOK();
    const uint64_t cold_delta = cold_now >= g_perf_last_cold_us ? cold_now - g_perf_last_cold_us : 0;
    ps.cold_us += cold_delta;
    g_perf_last_cold_us = cold_now;

    const uint64_t cold_mmid_now = MOE_MMID_TIMER_HOOK();
    const uint64_t cold_mmid_delta = cold_mmid_now >= g_perf_last_cold_mmid_us ?
        cold_mmid_now - g_perf_last_cold_mmid_us : 0;
    ps.cold_mmid_us += cold_mmid_delta;
    g_perf_last_cold_mmid_us = cold_mmid_now;

    const uint64_t h2d_delta = g_h2d_us >= g_perf_last_h2d_us ? g_h2d_us - g_perf_last_h2d_us : 0;
    const uint64_t h2d_bytes_delta = g_h2d_bytes >= g_perf_last_h2d_bytes ? g_h2d_bytes - g_perf_last_h2d_bytes : 0;
    const uint64_t h2d_calls_delta = g_h2d_calls >= g_perf_last_h2d_calls ? g_h2d_calls - g_perf_last_h2d_calls : 0;
    ps.h2d_us += h2d_delta;
    ps.h2d_bytes += h2d_bytes_delta;
    ps.h2d_calls += h2d_calls_delta;
    if (g_h2d_slot_stalls >= g_perf_last_h2d_slot_stalls) {
        ps.h2d_slot_stalls += g_h2d_slot_stalls - g_perf_last_h2d_slot_stalls;
    }
    if (g_h2d_slot_stall_us >= g_perf_last_h2d_slot_stall_us) {
        ps.h2d_slot_stall_us += g_h2d_slot_stall_us - g_perf_last_h2d_slot_stall_us;
    }
    if (g_h2d_async_bytes >= g_perf_last_h2d_async_bytes) {
        ps.h2d_async_bytes += g_h2d_async_bytes - g_perf_last_h2d_async_bytes;
    }
    if (g_h2d_async_calls >= g_perf_last_h2d_async_calls) {
        ps.h2d_async_calls += g_h2d_async_calls - g_perf_last_h2d_async_calls;
    }
    if (g_h2d_sync_fallback_bytes >= g_perf_last_h2d_sync_fallback_bytes) {
        ps.h2d_sync_fallback_bytes += g_h2d_sync_fallback_bytes - g_perf_last_h2d_sync_fallback_bytes;
    }
    if (g_h2d_sync_fallback_calls >= g_perf_last_h2d_sync_fallback_calls) {
        ps.h2d_sync_fallback_calls += g_h2d_sync_fallback_calls - g_perf_last_h2d_sync_fallback_calls;
    }
    const uint64_t publish_delta = g_h2d_publish_us >= g_perf_last_h2d_publish_us ?
        g_h2d_publish_us - g_perf_last_h2d_publish_us : 0;
    ps.h2d_publish_us += publish_delta;
    if (g_h2d_publish_calls >= g_perf_last_h2d_publish_calls) {
        ps.h2d_publish_calls += g_h2d_publish_calls - g_perf_last_h2d_publish_calls;
    }
    g_perf_last_h2d_us = g_h2d_us;
    g_perf_last_h2d_bytes = g_h2d_bytes;
    g_perf_last_h2d_calls = g_h2d_calls;
    g_perf_last_h2d_slot_stalls = g_h2d_slot_stalls;
    g_perf_last_h2d_slot_stall_us = g_h2d_slot_stall_us;
    g_perf_last_h2d_async_bytes = g_h2d_async_bytes;
    g_perf_last_h2d_async_calls = g_h2d_async_calls;
    g_perf_last_h2d_sync_fallback_bytes = g_h2d_sync_fallback_bytes;
    g_perf_last_h2d_sync_fallback_calls = g_h2d_sync_fallback_calls;
    g_perf_last_h2d_publish_us = g_h2d_publish_us;
    g_perf_last_h2d_publish_calls = g_h2d_publish_calls;

    // Passive dispatcher evidence only. The hot GPU signal is deliberately a
    // graph-wait-per-hot-route proxy because CPU cold work can overlap GPU work;
    // subtracting cold time from graph wait would fabricate an isolated GPU timer.
    if (route_hot > 0) {
        dispatcher_cost_update(ps.dispatcher_hot_gpu_route_us, (double) wait_us/(double) route_hot);
    }
    if (route_cold > 0 && cold_mmid_delta > 0) {
        dispatcher_cost_update(ps.dispatcher_cold_cpu_mmid_route_us, (double) cold_mmid_delta/(double) route_cold);
    }
    dispatcher_cost_update(ps.dispatcher_graph_publish_wait_us, (double) wait_us + (double) publish_delta);
    const uint64_t h2d_submit_delta = h2d_delta >= publish_delta ? h2d_delta - publish_delta : h2d_delta;
    if (h2d_bytes_delta > 0) {
        dispatcher_cost_update(ps.dispatcher_h2d_us_per_mib,
                (double) h2d_submit_delta * (1024.0*1024.0) / (double) h2d_bytes_delta);
    }
}

size_t expert_weight_bytes(const llama_model & model) {
    size_t b = 0;
    for (int il = 0; il < (int) model.hparams.n_layer(); il++) {
        const llama_layer & l = model.layers[il];
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (w) {
                b += ggml_nbytes(w);
            }
        }
    }
    return b;
}

void init(const llama_model & model) {
    std::lock_guard<std::mutex> init_lock(g_init_mu);
    const llama_model * owner = g_owner_model.load(std::memory_order_acquire);
    if (owner == &model) {
        g_tier_disabled.store(true, std::memory_order_release);
        if (!g_tier_conflict_logged) {
            TIER_LOG("%s: expert tiering disabled (multiple contexts share one process-global tier)\n", __func__);
            g_tier_conflict_logged = true;
        }
        return;
    }
    if (owner || g_tier_disabled.load(std::memory_order_acquire)) {
        if (!g_tier_conflict_logged) {
            TIER_LOG("%s: expert tiering disabled (tier already belongs to another model)\n", __func__);
            g_tier_conflict_logged = true;
        }
        return;
    }
    if (!g_layers.empty()) {
        TIER_LOG("%s: expert tiering disabled (incomplete global tier state)\n", __func__);
        g_tier_disabled.store(true, std::memory_order_release);
        return;
    }
    g_stats_enabled = getenv("LLAMA_EXPERT_STATS") != nullptr;
    const char * env_s   = getenv("LLAMA_EXPERT_S");
    const char * env_hot = getenv("LLAMA_EXPERT_HOT");
    if (const char * e = getenv("LLAMA_EXPERT_ADAPT")) {
        g_adapt = atoi(e) != 0;
    } else {
        g_adapt = true; // Auto-fit and online adaptation ON by default
    }
    if (const char * e = getenv("LLAMA_EXPERT_MADVISE")) {
        g_madvise = atoi(e) != 0;
    }
    if (const char * e = getenv("LLAMA_EXPERT_RAMPOOL")) {
        g_pool_bytes = (size_t) (atof(e)*1073741824.0);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREDICT")) {
        g_predict = atoi(e) != 0;
    }
    if (const char * e = getenv("LLAMA_EXPERT_MARKOV")) {
        g_markov_predict = atoi(e) != 0;
    }
    if (const char * e = getenv("LLAMA_EXPERT_MARKOV_TOP_N")) {
        g_markov_top_n = std::clamp(atoi(e), 1, 8);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREDICT_LOG")) {
        if (e[0]) {
            g_pred_log = fopen(e, "w");
            if (!g_pred_log) {
                TIER_LOG("%s: cannot open predict log '%s'\n", __func__, e);
            }
            std::string rpath(e);
            const auto dot = rpath.rfind('.');
            if (dot != std::string::npos) {
                rpath = rpath.substr(0, dot) + ".route" + rpath.substr(dot);
            } else {
                rpath += ".route";
            }
            g_route_log = fopen(rpath.c_str(), "w");
            if (!g_route_log) {
                TIER_LOG("%s: cannot open route trace '%s'\n", __func__, rpath.c_str());
            }
        }
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_MB")) {
        g_work_budget = (int64_t) (atof(e)*1048576.0);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_GB")) {
        g_poolB_bytes = (size_t) (atof(e)*1073741824.0);
    }
    if (const char * e = getenv("LLAMA_EXPERT_PREFETCH_THREADS")) {
        g_n_workers = std::min(8, std::max(1, atoi(e)));
    }
    const bool want_pread = [] {
        const char * e = getenv("LLAMA_EXPERT_PREAD");
        return !e || atoi(e) != 0;
    }();

    bool manual_S = false;
    if (env_s) {
        g_S = std::max(0, atoi(env_s));
        manual_S = (env_s != nullptr);
    }

    if (const char * e = getenv("LLAMA_EXPERT_TMAX")) {
        g_tmax = std::max(0, atoi(e));
    }
    if (const char * e = getenv("LLAMA_EXPERT_SCORE_SPLIT")) {
        g_score_split = atoi(e) != 0;
    }
    if (const char * e = getenv("LLAMA_EXPERT_SCORE_TG_WEIGHT")) {
        g_score_tg_weight = std::clamp((float) atof(e), 0.0f, 1.0f);
    }
    if (const char * e = getenv("LLAMA_EXPERT_LAYER_BUDGET")) {
        g_layer_budget_policy = atoi(e) != 0;
    }

    const bool want_h2d_stage = [] {
        const char * e = getenv("LLAMA_EXPERT_ASYNC_H2D");
        return e && atoi(e) != 0;
    }();
    size_t h2d_staging_bytes = 0;
    if (const char * e = getenv("LLAMA_EXPERT_H2D_STAGING_MB")) {
        h2d_staging_bytes = (size_t) (atof(e)*1048576.0);
    }

    const int n_layer  = model.hparams.n_layer();
    const int n_expert = model.hparams.n_expert;

    if (g_route_log) {
        tier_resolve_moe_hooks();
        MOE_ROUTE_HOOK(g_route_log, n_layer);
    }

    std::vector<std::vector<std::pair<int64_t, int32_t>>> heat;
    if (env_hot && env_hot[0] && !parse_heat_csv(env_hot, n_layer, heat)) {
        return;
    }

    // sidecar persistence: compute fingerprint, set path, load if no HOT override
    g_fingerprint = compute_fingerprint(model);
    const std::string mpath = model.path();
    if (!mpath.empty()) {
        g_sidecar_path = mpath + ".tier";
    }
    if (!env_hot || !env_hot[0]) {
        const auto sidecar_heat = load_sidecar(model);
        if (!sidecar_heat.empty()) {
            heat.resize(n_layer);
            for (int il = 0; il < n_layer && il < (int) sidecar_heat.size(); il++) {
                for (int e = 0; e < n_expert && e < (int) sidecar_heat[il].size(); e++) {
                    if (sidecar_heat[il][e] > 0.0f) {
                        heat[il].emplace_back((int64_t) sidecar_heat[il][e], e);
                    }
                }
            }
        }
    } else {
        TIER_LOG("%s: LLAMA_EXPERT_HOT set, ignoring sidecar\n", __func__);
    }

    if (getenv("LLAMA_EXPERT_STATS") || getenv("LLAMA_EXPERT_USAGE")) {
        atexit(dump_stats);
    }
    atexit(save_sidecar);

    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        TIER_LOG("%s: expert tiering disabled (no GPU device found)\n", __func__);
        return;
    }

    // pread demand fetch: init + KAT
    if (want_pread) {
        const std::string mpath = model.path();
        void * mbase = model.mmap_base();
        const ggml_tensor * kat_w = nullptr;
        for (int il = 0; il < n_layer && !kat_w; il++) {
            const llama_layer & l = model.layers[il];
            for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
                if (w && w->data && ggml_backend_buffer_is_host(w->buffer)) {
                    kat_w = w;
                    break;
                }
            }
        }
        if (mpath.empty()) {
            TIER_LOG("%s: pread disabled (no model path)\n", __func__);
            g_pread_disabled = true;
        } else if (!mbase) {
            TIER_LOG("%s: pread disabled (no mmap; --no-mmap?)\n", __func__);
            g_pread_disabled = true;
        } else if (!kat_w) {
            TIER_LOG("%s: pread disabled (no host expert weights)\n", __func__);
            g_pread_disabled = true;
        } else if (!tier_pread_init(mpath.c_str())) {
            TIER_LOG("%s: pread disabled (open failed)\n", __func__);
        } else if (!tier_pread_kat(kat_w->data, (size_t) ((const char *) kat_w->data - (const char *) mbase))) {
            TIER_LOG("%s: pread disabled (KAT failed)\n", __func__);
            tier_pread_close();
        } else {
            g_pread_mmap_base = (const char *) mbase;
            g_pread_mmap_size = model.mmap_size();
            TIER_LOG("%s: pread KAT passed\n", __func__);
            atexit(tier_pread_close);
        }
    }

    // Calculate per-expert slot memory size across all layers for Auto-Fit
    size_t bytes_per_slot_all_layers = 0;
    std::vector<size_t> layer_slot_bytes(n_layer, 0);
    int n_tensors = 0;
    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (w && ggml_backend_buffer_is_host(w->buffer) &&
                w->ne[3] == 1 && (int) w->ne[2] == n_expert) {
                n_tensors++;
                if (n_expert > 0) {
                    const size_t bytes = ggml_nbytes(w) / n_expert;
                    bytes_per_slot_all_layers += bytes;
                    layer_slot_bytes[il] += bytes;
                }
            }
        }
    }

    if (n_tensors == 0) {
        TIER_LOG("%s: expert tiering disabled (no host expert weights found)\n", __func__);
        return;
    }

    // Reserve 512 MB for the CUDA runtime, display, alignment and the graph
    // capture buffers that allocate after this sizing (300 MB let capture OOM)
    const size_t safety_buffer = 512ULL * 1024 * 1024;
    size_t free_vram = 0, total_vram = 0;
    ggml_backend_dev_memory(dev, &free_vram, &total_vram);
    const size_t usable_vram = (free_vram > safety_buffer) ? (free_vram - safety_buffer) : 0;

    if (!manual_S && bytes_per_slot_all_layers > 0) {
        if (usable_vram >= bytes_per_slot_all_layers) {
            int autofit_s = (int) (usable_vram / bytes_per_slot_all_layers);
            g_S = std::clamp(autofit_s, 1, n_expert);
            TIER_LOG("%s: AUTO-FIT ENGINE -> set S = %d (Free VRAM: %.2f / %.2f GB, per-slot size: %.2f MB, total experts: %d)\n",
                           __func__, g_S, (double)free_vram / (1024.0*1024.0*1024.0),
                           (double)total_vram / (1024.0*1024.0*1024.0),
                           (double)bytes_per_slot_all_layers / (1024.0*1024.0), n_expert);
        } else {
            g_S = 0;
            TIER_LOG("%s: AUTO-FIT ENGINE -> Insufficient free VRAM (%.2f MB free vs %.2f MB required per slot), expert tiering on GPU disabled\n",
                           __func__, (double)free_vram / (1024.0*1024.0), (double)bytes_per_slot_all_layers / (1024.0*1024.0));
        }
    }

    // clamp a forced S to what fits; the unclamped path OOMs at graph capture
    if (manual_S && bytes_per_slot_all_layers > 0) {
        const int afford = (int) (usable_vram / bytes_per_slot_all_layers);
        if (g_S > afford) {
            TIER_LOG("%s: manual S = %d exceeds free VRAM, clamping to %d\n", __func__, g_S, afford);
            g_S = std::max(afford, 0);
        }
    }

    if (g_S <= 0) {
        TIER_LOG("%s: expert tiering disabled (S=0)\n", __func__);
        return;
    }

    std::vector<int> layer_slots(n_layer, g_S);
    if (g_layer_budget_policy) {
        std::vector<hot_slot_policy_layer> policy_layers(n_layer);
        for (int il = 0; il < n_layer; il++) {
            policy_layers[il].expert_bytes = layer_slot_bytes[il];
            policy_layers[il].scores.assign(n_expert, 0.0f);
            if (il < (int) heat.size()) {
                for (const auto & p : heat[il]) {
                    if (p.second >= 0 && p.second < n_expert) {
                        policy_layers[il].scores[p.second] = std::max(policy_layers[il].scores[p.second], (float) p.first);
                    }
                }
            }
        }
        const size_t byte_budget = bytes_per_slot_all_layers > SIZE_MAX/(size_t) g_S ? 0 :
                bytes_per_slot_all_layers*(size_t) g_S;
        layer_slots = hot_slot_budget_policy(policy_layers, byte_budget, g_S);
        TIER_LOG("%s: per-layer hot-slot policy on (budget %.2f MiB)\n", __func__,
                (double) byte_budget/(1024.0*1024.0));
    }

    TIER_LOG("%s: page hints: %s\n", __func__, g_madvise ? "on" : "off");
    ggml_backend_buffer_type_t buft_gpu = ggml_backend_dev_buffer_type(dev);
    ggml_backend_buffer_type_t buft_cpu = ggml_backend_cpu_buffer_type();

    struct ggml_init_params ip_gpu = {
        /*.mem_size   =*/ ggml_tensor_overhead()*2*n_tensors + 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_init_params ip_cpu = {
        /*.mem_size   =*/ ggml_tensor_overhead()*3*n_tensors + 1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    g_ctx_gpu = ggml_init(ip_gpu);
    g_ctx_cpu = ggml_init(ip_cpu);

    std::vector<int32_t> lut_host(n_expert);
    std::vector<int32_t> mask_host(n_expert);
    std::vector<char>    zeros;

    size_t total_bytes = 0;
    int64_t hits = 0, total = 0;

    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];

        // top-S experts of this layer by heat
        std::vector<int32_t> top;
        if (il < (int) heat.size() && !heat[il].empty()) {
            auto h = heat[il];
            std::sort(h.begin(), h.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            int64_t layer_total = 0;
            for (const auto & p : h) {
                layer_total += p.first;
            }
            for (int i = 0; i < (int) h.size() && i < layer_slots[il]; i++) {
                top.push_back(h[i].second);
                hits += h[i].first;
            }
            total += layer_total;
        }
        if (top.empty()) {
            if (!g_adapt) {
                continue;
            }
            for (int i = 0; i < std::min(layer_slots[il], n_expert); i++) {
                top.push_back(i);
            }
        }

        std::fill(lut_host.begin(),  lut_host.end(),  (int32_t) top.size());
        std::fill(mask_host.begin(), mask_host.end(), 1);
        for (int s = 0; s < (int) top.size(); s++) {
            lut_host[top[s]]  = s;
            mask_host[top[s]] = 0;
        }

        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            if (!w || !ggml_backend_buffer_is_host(w->buffer)) {
                continue;
            }
            if (w->ne[3] != 1 || (int) w->ne[2] != n_expert) {
                continue;
            }
            const size_t slice = ggml_nbytes(w)/n_expert;

            store s;
            s.w_hot  = ggml_new_tensor_3d(g_ctx_gpu, w->type, w->ne[0], w->ne[1], g_adapt ? layer_slots[il] + 1 : (int) top.size() + 1);
            s.lut    = ggml_new_tensor_2d(g_ctx_gpu, GGML_TYPE_I32, 1, n_expert);
            s.mask   = ggml_new_tensor_1d(g_ctx_cpu, GGML_TYPE_I32, n_expert);
            s.counts = ggml_new_tensor_1d(g_ctx_cpu, GGML_TYPE_I32, n_expert + 1);
            s.il      = il;
            s.is_down = (w == l.ffn_down_exps);
            s.poolable    = w->data && model.weights_discardable(w->data);
            s.discardable = g_madvise && s.poolable;
            s.ptrs = g_pool_bytes > 0 ? ggml_new_tensor_1d(g_ctx_cpu, GGML_TYPE_I64, n_expert) : nullptr;

            ggml_set_name(s.w_hot, (std::string(w->name) + ".hot").c_str());

            // tensors are allocated lazily below per-context; collect first
            // (allocation happens after all tensors are created)
            g_stores[w] = s;
            total_bytes += ggml_nbytes(s.w_hot);
            (void) slice;
        }

        // per-layer tensors are ready; fill them after global allocation
    }

    // allocate all tensors
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors_from_buft(g_ctx_gpu, buft_gpu);
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors_from_buft(g_ctx_cpu, buft_cpu);
    if (!buf_gpu) {
        TIER_LOG("%s: expert tiering GPU allocation failed (VRAM full), falling back to CPU host execution\n", __func__);
        return;
    }
    ggml_backend_buffer_set_usage(buf_gpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    if (buf_cpu) {
        ggml_backend_buffer_set_usage(buf_cpu, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // Promotion is called after scheduler synchronization. update() receives
    // the scheduler's same-device backend and queues its waits there; a
    // missing or mismatched backend falls back to a local copy wait.
    if (want_h2d_stage && h2d_staging_bytes >= 3) {
        g_h2d_compute_backend = ggml_backend_dev_init(dev, nullptr);
        if (g_h2d_compute_backend) {
            g_h2d_stage = h2d_stage_create(g_h2d_compute_backend, h2d_staging_bytes/3);
        }
        if (!h2d_stage_is_async(g_h2d_stage)) {
            tier_h2d_stage_shutdown();
            TIER_LOG("%s: async H2D promotion disabled (backend staging unavailable)\n", __func__);
        } else {
            atexit(tier_h2d_stage_shutdown);
            TIER_LOG("%s: async H2D promotion staging: 3 x %.2f MiB\n", __func__,
                    (double) h2d_stage_capacity(g_h2d_stage)/(1024.0*1024.0));
        }
    } else if (want_h2d_stage) {
        TIER_LOG("%s: async H2D promotion disabled (LLAMA_EXPERT_H2D_STAGING_MB must be at least 3)\n", __func__);
    }

    // fill lut/mask/hot weights
    g_layers.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        const llama_layer & l = model.layers[il];

        std::vector<int32_t> top;
        if (il < (int) heat.size() && !heat[il].empty()) {
            auto h = heat[il];
            std::sort(h.begin(), h.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
            for (int i = 0; i < (int) h.size() && i < layer_slots[il]; i++) {
                top.push_back(h[i].second);
            }
        }
        if (top.empty()) {
            if (!g_adapt) {
                continue;
            }
            for (int i = 0; i < std::min(layer_slots[il], n_expert); i++) {
                top.push_back(i);
            }
        }

        std::fill(lut_host.begin(),  lut_host.end(),  (int32_t) top.size());
        std::fill(mask_host.begin(), mask_host.end(), 1);
        for (int s = 0; s < (int) top.size(); s++) {
            lut_host[top[s]]  = s;
            mask_host[top[s]] = 0;
        }

        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            auto it = g_stores.find(w);
            if (it == g_stores.end()) {
                continue;
            }
            store & s = it->second;
            const size_t slice = ggml_nbytes(w)/n_expert;

            ggml_backend_tensor_set(s.lut,  lut_host.data(),  0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(s.mask, mask_host.data(), 0, n_expert*sizeof(int32_t));
            if (s.ptrs) {
                for (int e = 0; e < n_expert; e++) {
                    ((int64_t *) s.ptrs->data)[e] = (int64_t) (uintptr_t) ((const char *) w->data + slice*e);
                }
            }

            if (zeros.size() < slice) {
                zeros.assign(slice, 0);
            }
            if (s.w_hot && s.w_hot->buffer) {
                ggml_backend_tensor_set(s.w_hot, zeros.data(), slice*top.size(), slice);
                for (int sl = 0; sl < (int) top.size(); sl++) {
                    const void * src_ptr = w->data ? (const char *) w->data + slice*top[sl] : zeros.data();
                    ggml_backend_tensor_set(s.w_hot, src_ptr, slice*sl, slice);
                    if (s.discardable && w->data) {
                        tier_madvise(src_ptr, slice, true);
                    }
                }
            }
        }

        // group the stores of this layer for stats/repin
        layer_tier L;
        L.il       = il;
        L.n_expert = n_expert;
        L.sentinel = (int) top.size();
        L.n_slots  = g_adapt ? layer_slots[il] + 1 : (int) top.size() + 1;
        L.slot_expert.assign(L.n_slots, -1);
        L.hot_generation.assign(L.n_slots, 1);
        L.dwell.assign(L.n_slots, 0);
        L.lut_host = lut_host;
        L.score.assign(n_expert, 0.0f);
        L.score_pp.assign(n_expert, 0.0f);
        L.score_tg.assign(n_expert, 0.0f);
        L.cum.assign(n_expert, 0);
        L.cum_pp.assign(n_expert, 0);
        L.cum_tg.assign(n_expert, 0);
        L.markov.resize(n_expert);
        L.markov_prev.assign(n_expert, 0);
        L.markov_pending.assign(n_expert, 0);
        if (il < (int) heat.size()) {
            for (const auto & p : heat[il]) {
                L.cum[p.second]   = (uint64_t) p.first;
                L.score[p.second] = (float) p.first; // warm start from seed
                L.score_pp[p.second] = (float) p.first;
                L.score_tg[p.second] = (float) p.first;
            }
        }
        for (int s = 0; s < (int) top.size(); s++) {
            L.slot_expert[s] = top[s];
        }
        for (ggml_tensor * w : {l.ffn_gate_exps, l.ffn_up_exps, l.ffn_down_exps, l.ffn_gate_up_exps}) {
            auto it = g_stores.find(w);
            if (it == g_stores.end()) {
                continue;
            }
            memset(it->second.counts->data, 0, (n_expert + 1)*sizeof(int32_t));
            it->second.h2d_slot_identity.resize((size_t) L.n_slots);
            for (int s = 0; s < L.n_slots; s++) {
                uint64_t identity = g_h2d_next_identity++;
                if (identity == 0) {
                    identity = g_h2d_next_identity++;
                }
                it->second.h2d_slot_identity[(size_t) s] = identity;
            }
            L.ws.push_back({w, &it->second});
            if (it->second.is_down) {
                L.sd = &it->second;
            }
        }
        if (!L.ws.empty()) {
            g_layers[il] = std::move(L);
        }
    }

    // RAM pool: one block per layer; slot = one expert across all its tensors
    if (g_pool_bytes > 0) {
        int n_pool_layers = 0;
        for (auto & L : g_layers) {
            if (L.sd && L.sd->poolable) {
                n_pool_layers++;
            }
        }
        for (auto & L : g_layers) {
            if (!L.sd || !L.sd->poolable || n_pool_layers == 0) {
                continue;
            }
            size_t slot = 0;
            for (auto & kv : L.ws) {
                kv.second->pool_off = slot;
                slot += pool_align(ggml_nbytes(kv.first)/L.n_expert);
            }
            L.pool_slot_bytes = slot;
            const size_t per_layer = g_pool_bytes/(size_t) n_pool_layers;
            L.n_pool_slots = slot > 0 ? (int) (per_layer/slot) : 0;
            if (L.n_pool_slots <= 0) {
                continue;
            }
            const size_t bytes = (size_t) L.n_pool_slots*slot;
#if defined(_WIN32)
            L.pool = (char *) _aligned_malloc(bytes, POOL_ALIGN);
#else
            void * p = nullptr;
            if (posix_memalign(&p, POOL_ALIGN, bytes) != 0) {
                p = nullptr;
            }
            L.pool = (char *) p;
#endif
            if (!L.pool) {
                TIER_LOG("%s: pool alloc failed for layer %d (%zu MiB)\n", __func__, L.il, bytes >> 20);
                L.n_pool_slots = 0;
                continue;
            }
            g_pool_alloc += bytes;
            L.pool_slot_expert.assign(L.n_pool_slots, -1);
            L.pool_dwell.assign(L.n_pool_slots, 0);
            L.pool_slot_state.reset(new std::atomic<int32_t>[L.n_pool_slots]);
            for (int k = 0; k < L.n_pool_slots; k++) {
                L.pool_slot_state[k].store(0, std::memory_order_relaxed);
            }
            L.pool_lut.reset(new std::atomic<int32_t>[L.n_expert]);
            for (int e = 0; e < L.n_expert; e++) {
                L.pool_lut[e].store(-1, std::memory_order_relaxed);
            }
            if (!heat.empty()) {
                // seed: hottest cold experts by score
                std::vector<int32_t> cold;
                for (int e = 0; e < L.n_expert; e++) {
                    if (L.lut_host[e] == L.sentinel) {
                        cold.push_back(e);
                    }
                }
                std::sort(cold.begin(), cold.end(), [&](int32_t a, int32_t b) {
                    return placement_score(L, a) != placement_score(L, b) ? placement_score(L, a) > placement_score(L, b) : a < b;
                });
                for (int k = 0; k < L.n_pool_slots && k < (int) cold.size(); k++) {
                    pool_fill(L, k, cold[k]);
                }
            }
        }
        TIER_LOG("%s: expert pool: %.2f GiB over %d layers\n",
                __func__, (double) g_pool_alloc/(double) (1 << 30), n_pool_layers);
    }

    // pread staging ring (LLAMA_EXPERT_PREAD): bounded, page-aligned, owned
    // by the cold path; sized to hold one token's routed experts of the
    // widest layer. requires the ptrs path (LLAMA_EXPERT_RAMPOOL > 0).
    if (want_pread && !g_pread_disabled) {
        if (g_pool_bytes == 0) {
            TIER_LOG("%s: pread ring disabled (needs LLAMA_EXPERT_RAMPOOL > 0)\n", __func__);
        } else {
            g_stage_fail_test = [] {
                const char * e = getenv("LLAMA_EXPERT_PREAD_FAIL");
                return e && atoi(e) != 0;
            }();
            size_t max_slot = 0;
            for (auto & L : g_layers) {
                if (L.sd && L.sd->poolable && L.pool_slot_bytes > 0) {
                    max_slot = std::max(max_slot, L.pool_slot_bytes);
                }
            }
            const size_t page = 4096;
            g_stage_stride = max_slot > 0 ? (max_slot + page - 1) & ~(page - 1) : 0;
            size_t ring_bytes = (size_t) model.hparams.n_expert_used*g_stage_stride;
            if (const char * e = getenv("LLAMA_EXPERT_PREAD_RING_MB")) {
                ring_bytes = (size_t) (atof(e)*1048576.0);
            }
            g_stage_n = g_stage_stride > 0 ? (int) std::max((size_t) 1, ring_bytes/g_stage_stride) : 0;
            // single-file guard: every poolable tensor must lie in the first
            // file mapping; sharded models disable the ring
            const char * mbase = (const char *) model.mmap_base();
            const size_t msize = model.mmap_size();
            if (g_stage_n > 0 && (!mbase || msize == 0)) {
                g_stage_n = 0;
            }
            for (const auto & L : g_layers) {
                if (g_stage_n == 0) {
                    break;
                }
                if (!L.sd || !L.sd->poolable || L.pool_slot_bytes == 0) {
                    continue;
                }
                for (const auto & kv : L.ws) {
                    const char * d = (const char *) kv.first->data;
                    if (d < mbase || (size_t) (d - mbase) + ggml_nbytes(kv.first) > msize) {
                        g_stage_n = 0;
                        TIER_LOG("%s: pread ring disabled (tensor outside first file mapping - sharded?)\n", __func__);
                        break;
                    }
                }
            }
            if (g_stage_n > 0) {
#if defined(_WIN32)
                g_stage_base = (char *) _aligned_malloc((size_t) g_stage_n*g_stage_stride, page);
#else
                void * p = nullptr;
                if (posix_memalign(&p, page, (size_t) g_stage_n*g_stage_stride) != 0) {
                    p = nullptr;
                }
                g_stage_base = (char *) p;
#endif
                if (!g_stage_base) {
                    TIER_LOG("%s: pread ring alloc failed (%d x %.2f MiB)\n", __func__,
                            g_stage_n, (double) g_stage_stride/(1024.0*1024.0));
                    g_stage_n = 0;
                } else {
                    g_stage_word.reset(new std::atomic<uint64_t>[g_stage_n]);
                    for (int k = 0; k < g_stage_n; k++) {
                        g_stage_word[k].store(0, std::memory_order_relaxed);
                    }
                    g_stage_mmap = mbase;
                    for (auto & L : g_layers) {
                        if (!L.sd || !L.sd->poolable || L.pool_slot_bytes == 0) {
                            continue;
                        }
                        for (auto & kv : L.ws) {
                            store & st = *kv.second;
                            if (!st.ptrs || !st.ptrs->data) {
                                continue;
                            }
                            g_stage_ix[st.ptrs->data] = stage_ctx{ L.il, (int64_t) st.pool_off,
                                    (const char *) kv.first->data, (int64_t) (ggml_nbytes(kv.first)/L.n_expert) };
                        }
                    }
                    tier_resolve_moe_hooks();
                    MOE_FETCH_HOOK(stage_fetch);
                    TIER_LOG("%s: pread ring: %d slots x %.2f MiB%s\n", __func__,
                            g_stage_n, (double) g_stage_stride/(1024.0*1024.0),
                            g_stage_fail_test ? " (FAIL-TEST)" : "");
                }
            }
        }
    }

    // spec pool (LLAMA_EXPERT_PREFETCH_GB): worker-only, probe-served; slots
    // reuse the pool-A layout (pool_slot_bytes, per-store pool_off)
    if (g_predict && g_poolB_bytes > 0) {
        int n_pool_layers = 0;
        for (auto & L : g_layers) {
            if (L.sd && L.sd->poolable) {
                n_pool_layers++;
            }
        }
        for (auto & L : g_layers) {
            if (!L.sd || !L.sd->poolable || n_pool_layers == 0) {
                continue;
            }
            const size_t per_layer = g_poolB_bytes/(size_t) n_pool_layers;
            L.n_slotsB = L.pool_slot_bytes > 0 ? (int) (per_layer/L.pool_slot_bytes) : 0;
            if (L.n_slotsB <= 0) {
                continue;
            }
            const size_t bytes = (size_t) L.n_slotsB*L.pool_slot_bytes;
#if defined(_WIN32)
            L.poolB = (char *) _aligned_malloc(bytes, POOL_ALIGN);
#else
            void * p = nullptr;
            if (posix_memalign(&p, POOL_ALIGN, bytes) != 0) {
                p = nullptr;
            }
            L.poolB = (char *) p;
#endif
            if (!L.poolB) {
                TIER_LOG("%s: spec pool alloc failed for layer %d (%zu MiB)\n", __func__, L.il, bytes >> 20);
                L.n_slotsB = 0;
                continue;
            }
            g_poolB_alloc += bytes;
            L.ownersB.assign(L.n_slotsB, -1);
            L.stateB.reset(new std::atomic<int32_t>[L.n_slotsB]);
            for (int k = 0; k < L.n_slotsB; k++) {
                L.stateB[k].store(0, std::memory_order_relaxed);
            }
            L.lutB.reset(new std::atomic<int32_t>[L.n_expert]);
            for (int e = 0; e < L.n_expert; e++) {
                L.lutB[e].store(-1, std::memory_order_relaxed);
            }
            L.lastB.reset(new std::atomic<int64_t>[L.n_slotsB]);
            for (int k = 0; k < L.n_slotsB; k++) {
                L.lastB[k].store(0, std::memory_order_relaxed);
            }
        }
        TIER_LOG("%s: spec pool: %.2f GiB over %d layers (%d threads)\n",
                __func__, (double) g_poolB_alloc/(double) (1 << 30), n_pool_layers, g_n_workers);
    }

                TIER_LOG("%s: expert tiering on: %d slots/layer, %zu tensors, %.2f GiB pinned, seed coverage %.1f%%\n",
            __func__, g_S, g_stores.size(), (double) total_bytes/(1 << 30),
            total > 0 ? 100.0*hits/total : 0.0);

    pred_init(model);
    g_owner_model.store(&model, std::memory_order_release);
}

void context_constructing(const llama_model & model) {
    std::lock_guard<std::mutex> init_lock(g_init_mu);
    const llama_model * owner = g_owner_model.load(std::memory_order_acquire);
    if (owner == &model) {
        // The tier is model-global and currently assumes a single live context
        // for its owner. Preserve the conservative behavior for a second
        // context of the same model.
        g_tier_disabled.store(true, std::memory_order_release);
        if (!g_tier_conflict_logged) {
            TIER_LOG("%s: expert tiering disabled (another owner context is constructing)\n", __func__);
            g_tier_conflict_logged = true;
        }
        return;
    }

    if (owner && owner != &model && !g_tier_conflict_logged) {
        // A different model (for example an MTP assistant) is safe: graph_scope
        // already gates tier hooks to g_owner_model, so this context simply uses
        // normal llama.cpp kernels without touching the owner's tier state.
        TIER_LOG("%s: foreign context detected; keeping expert tier active for owner model only\n", __func__);
        g_tier_conflict_logged = true;
    }
}

void model_destroying(const llama_model & model) {
    std::lock_guard<std::mutex> init_lock(g_init_mu);
    if (g_owner_model.load(std::memory_order_acquire) == &model) {
        g_owner_model.store(nullptr, std::memory_order_release);
        g_tier_disabled.store(true, std::memory_order_release);
    }
}

ggml_tensor * build_mul_mat_id(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x, ggml_tensor * ids) {
    if (!tier_graph_active()) {
        return ggml_mul_mat_id(ctx, w, x, ids);
    }

    const store * sp = nullptr;
    if (g_hot_only) {
        if (w == g_hot_moe.gate_w) {
            sp = g_hot_moe.gate;
        } else if (w == g_hot_moe.up_w) {
            sp = g_hot_moe.up;
        } else if (w == g_hot_moe.down_w) {
            sp = g_hot_moe.down;
        }
    }
    if (!sp) {
        auto it = g_stores.find(w);
        if (it == g_stores.end()) {
            return ggml_mul_mat_id(ctx, w, x, ids);
        }
        sp = &it->second;
    }

    const store & s = *sp;

    // get_rows wants matching trailing dims, so flatten ids to 1d first
    ggml_tensor * ids_flat = ggml_reshape_1d(ctx, ggml_cont(ctx, ids), ids->ne[0]*ids->ne[1]);
    ggml_tensor * hot_flat = ggml_get_rows(ctx, s.lut, ids_flat); // [1, n_used*n_tokens]
    ggml_tensor * ids_hot  = ggml_reshape_2d(ctx, hot_flat, ids->ne[0], ids->ne[1]);
    ggml_tensor * hot      = ggml_mul_mat_id(ctx, s.w_hot, x, ids_hot);    // GPU

    if (g_hot_only) {
        return hot;
    }

    ggml_tensor * cold     = ggml_mul_mat_id_cold(ctx, w, x, ids, s.mask, s.ptrs); // CPU

    return ggml_add(ctx, hot, cold);
}

bool begin_moe_cold(bool eligible,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * ids) {
    g_hot_only = false;
    g_hot_moe = {};
    if (!tier_graph_active() || !eligible || g_stores.empty()) {
        return false;
    }
    auto ig = g_stores.find(gate_w);
    auto iu = g_stores.find(up_w);
    auto id = g_stores.find(down_w);
    if (ig == g_stores.end() || iu == g_stores.end() || id == g_stores.end() ||
        ids->ne[1] > (int64_t) g_tmax) {
        return false;
    }
    g_hot_moe.gate_w = gate_w;
    g_hot_moe.up_w   = up_w;
    g_hot_moe.down_w = down_w;
    g_hot_moe.gate = &ig->second;
    g_hot_moe.up   = &iu->second;
    g_hot_moe.down = &id->second;
    g_hot_only = true;
    return true;
}

ggml_tensor * end_moe_cold(ggml_context * ctx,
        ggml_tensor * gate_w, ggml_tensor * up_w, ggml_tensor * down_w,
        ggml_tensor * x, ggml_tensor * ids) {
    if (!tier_graph_active() || !g_hot_only) {
        g_hot_only = false;
        g_hot_moe = {};
        return nullptr;
    }
    GGML_ASSERT(g_hot_moe.gate && g_hot_moe.up && g_hot_moe.down);
    GGML_ASSERT(g_hot_moe.gate_w == gate_w && g_hot_moe.up_w == up_w && g_hot_moe.down_w == down_w);
    return ggml_moe_cold(ctx, gate_w, up_w, down_w, x, ids, g_hot_moe.down->mask, g_hot_moe.down->counts,
            g_hot_moe.gate->ptrs, g_hot_moe.up->ptrs, g_hot_moe.down->ptrs);
}

ggml_tensor * build_moe_count(ggml_context * ctx, ggml_tensor * down_w, ggml_tensor * ids) {
    // Any non-fused layer needs the count op. Large batches use it alongside
    // the stock path, while generic tiered layers use it alongside cold MMID.
    if (!tier_graph_active() || g_hot_only || g_stores.empty()) {
        return nullptr;
    }
    auto it = g_stores.find(down_w);
    if (it == g_stores.end()) {
        return nullptr;
    }
    return ggml_moe_count(ctx, ids, it->second.counts);
}

}
