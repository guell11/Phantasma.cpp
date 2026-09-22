#include "tree-draft-sm89.h"

#include <algorithm>
#include <limits>

static_assert(COMMON_TREE_DRAFT_TENSOR_CORE_F16 == GGML_BACKEND_CUDA_TENSOR_CORE_F16, "tensor-core mode ABI mismatch");
static_assert(COMMON_TREE_DRAFT_TENSOR_CORE_BF16 == GGML_BACKEND_CUDA_TENSOR_CORE_BF16, "tensor-core mode ABI mismatch");
static_assert(COMMON_TREE_DRAFT_TENSOR_CORE_TF32 == GGML_BACKEND_CUDA_TENSOR_CORE_TF32, "tensor-core mode ABI mismatch");
static_assert(COMMON_TREE_DRAFT_TENSOR_CORE_FP8 == GGML_BACKEND_CUDA_TENSOR_CORE_FP8, "tensor-core mode ABI mismatch");

template<typename T, typename U>
static T tree_draft_narrow_saturate(U value) {
    return (T) std::min<U>(value, std::numeric_limits<T>::max());
}

common_tree_draft_device_capabilities common_tree_draft_device_capabilities_normalize(
        const ggml_backend_cuda_device_capabilities & capabilities) {
    common_tree_draft_device_capabilities result;
    result.available         = true;
    result.cc_major          = tree_draft_narrow_saturate<uint8_t>(capabilities.cc_major);
    result.cc_minor          = tree_draft_narrow_saturate<uint8_t>(capabilities.cc_minor);
    result.warp_size         = tree_draft_narrow_saturate<uint16_t>(capabilities.warp_size);
    result.max_threads_sm    = tree_draft_narrow_saturate<uint16_t>(capabilities.max_threads_sm);
    result.max_threads_block = tree_draft_narrow_saturate<uint16_t>(capabilities.max_threads_block);
    result.regs_sm           = capabilities.regs_sm;
    result.smem_sm           = capabilities.smem_sm;
    result.l2_bytes          = capabilities.l2_bytes;
    result.async_engines     = tree_draft_narrow_saturate<uint8_t>(capabilities.async_engines);
    result.tensor_core_modes = capabilities.tensor_core_modes;
    return result;
}

bool common_tree_draft_device_is_sm89(const common_tree_draft_device_capabilities & capabilities) {
    return capabilities.available && capabilities.cc_major == 8 && capabilities.cc_minor == 9;
}

common_tree_draft_launch_geometry_candidates common_tree_draft_sm89_launch_geometry_candidates(
        const common_tree_draft_device_capabilities & capabilities) {
    common_tree_draft_launch_geometry_candidates result;
    if (!common_tree_draft_device_is_sm89(capabilities) || capabilities.warp_size == 0 ||
        capabilities.max_threads_sm == 0 || capabilities.max_threads_block == 0) {
        return result;
    }

    static constexpr uint16_t block_sizes[] = { 64, 128, 256, 512 };
    for (uint16_t block_size : block_sizes) {
        if (block_size > capabilities.max_threads_block || block_size > capabilities.max_threads_sm ||
            block_size % capabilities.warp_size != 0) {
            continue;
        }

        const uint32_t resident_blocks = capabilities.max_threads_sm / block_size;
        const uint32_t warps_per_block = block_size / capabilities.warp_size;
        const uint32_t hardware_warps = capabilities.max_threads_sm / capabilities.warp_size;
        const uint32_t active_warps = std::min(hardware_warps, resident_blocks * warps_per_block);
        if (resident_blocks == 0 || active_warps == 0) {
            continue;
        }

        auto & candidate = result.values[result.count++];
        candidate.block_size = block_size;
        candidate.warps_per_block = tree_draft_narrow_saturate<uint16_t>(warps_per_block);
        candidate.resident_blocks_by_threads = tree_draft_narrow_saturate<uint16_t>(resident_blocks);
        candidate.active_warps_by_threads = tree_draft_narrow_saturate<uint16_t>(active_warps);
    }

    result.available = result.count != 0;
    return result;
}

common_tree_draft_occupancy_result common_tree_draft_sm89_occupancy(
        const common_tree_draft_device_capabilities & capabilities,
        const common_tree_draft_occupancy_request & request) {
    common_tree_draft_occupancy_result result;
    if (!common_tree_draft_device_is_sm89(capabilities) || request.block_size == 0 || request.max_blocks_sm == 0 ||
        capabilities.warp_size == 0 || capabilities.max_threads_sm == 0 ||
        request.block_size > capabilities.max_threads_block || request.block_size > capabilities.max_threads_sm ||
        request.block_size % capabilities.warp_size != 0) {
        return result;
    }

    const uint64_t shared_bytes = (uint64_t) request.static_smem_bytes + request.dynamic_smem_bytes;
    const uint64_t registers_per_block = (uint64_t) request.registers_per_thread * request.block_size;

    const uint32_t by_blocks = request.max_blocks_sm;
    const uint32_t by_threads = capabilities.max_threads_sm / request.block_size;
    const uint32_t by_registers = registers_per_block == 0 ? std::numeric_limits<uint32_t>::max() :
        (uint32_t) std::min<uint64_t>(capabilities.regs_sm / registers_per_block, std::numeric_limits<uint32_t>::max());
    const uint32_t by_smem = shared_bytes == 0 ? std::numeric_limits<uint32_t>::max() :
        (uint32_t) std::min<uint64_t>(capabilities.smem_sm / shared_bytes, std::numeric_limits<uint32_t>::max());

    uint32_t resident_blocks = by_blocks;
    common_tree_draft_occupancy_limit limiting_resource = COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_BLOCKS;
    if (by_threads < resident_blocks) {
        resident_blocks = by_threads;
        limiting_resource = COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_THREADS;
    }
    if (by_registers < resident_blocks) {
        resident_blocks = by_registers;
        limiting_resource = COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_REGISTERS;
    }
    if (by_smem < resident_blocks) {
        resident_blocks = by_smem;
        limiting_resource = COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_SHARED_MEMORY;
    }

    const uint32_t warps_per_block = request.block_size / capabilities.warp_size;
    const uint32_t resident_warps = resident_blocks * warps_per_block;
    result.available = true;
    result.resident_blocks = tree_draft_narrow_saturate<uint16_t>(resident_blocks);
    result.resident_warps = tree_draft_narrow_saturate<uint16_t>(resident_warps);
    result.occupancy = (float) ((double) resident_blocks * request.block_size / capabilities.max_threads_sm);
    result.limiting_resource = limiting_resource;
    return result;
}

bool common_tree_draft_vector_access_is_aligned(
        const common_tree_draft_vector_access_request & request,
        uint32_t vector_bytes) {
    if (request.element_size == 0 || request.element_count == 0 || vector_bytes == 0 ||
        vector_bytes % request.element_size != 0) {
        return false;
    }

    const uint64_t elements_per_vector = vector_bytes / request.element_size;
    return request.address % vector_bytes == 0 && request.stride_bytes % vector_bytes == 0 &&
           request.element_count % elements_per_vector == 0;
}

common_tree_draft_vector_access_result common_tree_draft_sm89_vector_access(
        const common_tree_draft_device_capabilities & capabilities,
        const common_tree_draft_vector_access_request & request) {
    common_tree_draft_vector_access_result result;
    if (request.element_size == 0 || request.element_count == 0 || request.element_size > UINT8_MAX) {
        return result;
    }

    result.available = true;
    result.vector_bytes = (uint8_t) request.element_size;
    result.elements_per_vector = 1;

    if (!common_tree_draft_device_is_sm89(capabilities)) {
        return result;
    }

    static constexpr uint8_t vector_widths[] = { 16, 8, 4 };
    for (uint8_t vector_bytes : vector_widths) {
        if (!common_tree_draft_vector_access_is_aligned(request, vector_bytes)) {
            continue;
        }

        result.vectorized = vector_bytes > request.element_size;
        result.vector_bytes = vector_bytes;
        result.elements_per_vector = (uint8_t) (vector_bytes / request.element_size);
        return result;
    }

    return result;
}

common_tree_draft_device_capability_cache::common_tree_draft_device_capability_cache(
        ggml_backend_cuda_get_device_capabilities_t probe) : probe_(probe) {
}

common_tree_draft_device_capabilities common_tree_draft_device_capability_cache::get(ggml_backend_dev_t device) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cache_.find(device);
    if (it != cache_.end()) {
        return it->second;
    }

    common_tree_draft_device_capabilities result;
    ggml_backend_cuda_device_capabilities raw = {};
    ggml_backend_cuda_get_device_capabilities_t probe = probe_;

    if (!probe && device) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(device);
        if (reg) {
            probe = (ggml_backend_cuda_get_device_capabilities_t) ggml_backend_reg_get_proc_address(
                    reg, "ggml_backend_cuda_get_device_capabilities");
        }
    }

    if (probe && probe(device, &raw)) {
        result = common_tree_draft_device_capabilities_normalize(raw);
    }

    cache_.emplace(device, result);
    return result;
}

common_tree_draft_device_capabilities common_tree_draft_device_capabilities_get(ggml_backend_dev_t device) {
    static common_tree_draft_device_capability_cache cache;
    return cache.get(device);
}
