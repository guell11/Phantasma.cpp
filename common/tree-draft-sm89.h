#pragma once

#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <unordered_map>

enum common_tree_draft_tensor_core_mode : uint32_t {
    COMMON_TREE_DRAFT_TENSOR_CORE_F16  = 1u << 0,
    COMMON_TREE_DRAFT_TENSOR_CORE_BF16 = 1u << 1,
    COMMON_TREE_DRAFT_TENSOR_CORE_TF32 = 1u << 2,
    COMMON_TREE_DRAFT_TENSOR_CORE_FP8  = 1u << 3,
};

struct common_tree_draft_device_capabilities {
    bool available = false;
    uint8_t cc_major = 0;
    uint8_t cc_minor = 0;
    uint16_t warp_size = 0;
    uint16_t max_threads_sm = 0;
    uint16_t max_threads_block = 0;
    uint32_t regs_sm = 0;
    uint64_t smem_sm = 0;
    uint64_t l2_bytes = 0;
    uint8_t async_engines = 0;
    uint32_t tensor_core_modes = 0;
};

common_tree_draft_device_capabilities common_tree_draft_device_capabilities_normalize(
        const ggml_backend_cuda_device_capabilities & capabilities);

bool common_tree_draft_device_is_sm89(const common_tree_draft_device_capabilities & capabilities);

struct common_tree_draft_launch_geometry_candidate {
    uint16_t block_size = 0;
    uint16_t warps_per_block = 0;
    uint16_t resident_blocks_by_threads = 0;
    uint16_t active_warps_by_threads = 0;
};

struct common_tree_draft_launch_geometry_candidates {
    bool available = false;
    uint8_t count = 0;
    std::array<common_tree_draft_launch_geometry_candidate, 4> values = {};
};

common_tree_draft_launch_geometry_candidates common_tree_draft_sm89_launch_geometry_candidates(
        const common_tree_draft_device_capabilities & capabilities);

enum common_tree_draft_occupancy_limit : uint8_t {
    COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_NONE = 0,
    COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_BLOCKS,
    COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_THREADS,
    COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_REGISTERS,
    COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_SHARED_MEMORY,
};

struct common_tree_draft_occupancy_request {
    uint16_t block_size = 0;
    uint16_t registers_per_thread = 0;
    uint32_t static_smem_bytes = 0;
    uint32_t dynamic_smem_bytes = 0;
    uint16_t max_blocks_sm = 0;
};

struct common_tree_draft_occupancy_result {
    bool available = false;
    uint16_t resident_blocks = 0;
    uint16_t resident_warps = 0;
    float occupancy = 0.0f;
    common_tree_draft_occupancy_limit limiting_resource = COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_NONE;
};

common_tree_draft_occupancy_result common_tree_draft_sm89_occupancy(
        const common_tree_draft_device_capabilities & capabilities,
        const common_tree_draft_occupancy_request & request);

struct common_tree_draft_vector_access_request {
    uint64_t address = 0;
    uint64_t stride_bytes = 0;
    uint64_t element_count = 0;
    uint32_t element_size = 0;
};

struct common_tree_draft_vector_access_result {
    bool available = false;
    bool vectorized = false;
    uint8_t vector_bytes = 0;
    uint8_t elements_per_vector = 0;
};

bool common_tree_draft_vector_access_is_aligned(
        const common_tree_draft_vector_access_request & request,
        uint32_t vector_bytes);

common_tree_draft_vector_access_result common_tree_draft_sm89_vector_access(
        const common_tree_draft_device_capabilities & capabilities,
        const common_tree_draft_vector_access_request & request);

class common_tree_draft_device_capability_cache {
public:
    explicit common_tree_draft_device_capability_cache(ggml_backend_cuda_get_device_capabilities_t probe = nullptr);

    common_tree_draft_device_capabilities get(ggml_backend_dev_t device);

private:
    ggml_backend_cuda_get_device_capabilities_t probe_;
    std::mutex mutex_;
    std::unordered_map<ggml_backend_dev_t, common_tree_draft_device_capabilities> cache_;
};

common_tree_draft_device_capabilities common_tree_draft_device_capabilities_get(ggml_backend_dev_t device);
