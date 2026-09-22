#include "tree-draft-sm89.h"

#include <cassert>
#include <cstdint>
#include <limits>

static int g_probe_calls = 0;
static int g_failed_probe_calls = 0;

static bool fake_sm89_probe(ggml_backend_dev_t device, ggml_backend_cuda_device_capabilities * capabilities) {
    g_probe_calls++;
    if (!device || !capabilities) {
        return false;
    }
    *capabilities = {
        8,
        9,
        32,
        1536,
        1024,
        65536,
        102400,
        50331648,
        2,
        GGML_BACKEND_CUDA_TENSOR_CORE_F16 |
            GGML_BACKEND_CUDA_TENSOR_CORE_BF16 |
            GGML_BACKEND_CUDA_TENSOR_CORE_TF32 |
            GGML_BACKEND_CUDA_TENSOR_CORE_FP8,
    };
    return true;
}

static bool fake_failed_probe(ggml_backend_dev_t, ggml_backend_cuda_device_capabilities *) {
    g_failed_probe_calls++;
    return false;
}

int main() {
    const ggml_backend_cuda_device_capabilities raw_sm89 = {
        8,
        9,
        32,
        1536,
        1024,
        65536,
        102400,
        50331648,
        2,
        GGML_BACKEND_CUDA_TENSOR_CORE_F16 |
            GGML_BACKEND_CUDA_TENSOR_CORE_BF16 |
            GGML_BACKEND_CUDA_TENSOR_CORE_TF32 |
            GGML_BACKEND_CUDA_TENSOR_CORE_FP8,
    };
    const auto sm89 = common_tree_draft_device_capabilities_normalize(raw_sm89);
    assert(sm89.available);
    assert(common_tree_draft_device_is_sm89(sm89));
    assert(sm89.warp_size == 32);
    assert(sm89.max_threads_sm == 1536);
    assert(sm89.max_threads_block == 1024);
    assert(sm89.regs_sm == 65536);
    assert(sm89.smem_sm == 102400);
    assert(sm89.l2_bytes == 50331648);
    assert(sm89.async_engines == 2);
    assert((sm89.tensor_core_modes & COMMON_TREE_DRAFT_TENSOR_CORE_FP8) != 0);

    const auto geometry = common_tree_draft_sm89_launch_geometry_candidates(sm89);
    assert(geometry.available);
    assert(geometry.count == 4);
    assert(geometry.values[0].block_size == 64);
    assert(geometry.values[0].warps_per_block == 2);
    assert(geometry.values[0].resident_blocks_by_threads == 24);
    assert(geometry.values[0].active_warps_by_threads == 48);
    assert(geometry.values[1].block_size == 128);
    assert(geometry.values[1].resident_blocks_by_threads == 12);
    assert(geometry.values[2].block_size == 256);
    assert(geometry.values[2].resident_blocks_by_threads == 6);
    assert(geometry.values[3].block_size == 512);
    assert(geometry.values[3].resident_blocks_by_threads == 3);

    auto limited_sm89 = sm89;
    limited_sm89.max_threads_block = 256;
    const auto limited_geometry = common_tree_draft_sm89_launch_geometry_candidates(limited_sm89);
    assert(limited_geometry.available);
    assert(limited_geometry.count == 3);
    assert(limited_geometry.values[2].block_size == 256);

    auto invalid_warp = sm89;
    invalid_warp.warp_size = 0;
    assert(!common_tree_draft_sm89_launch_geometry_candidates(invalid_warp).available);

    common_tree_draft_occupancy_request occupancy_request = {};
    occupancy_request.block_size = 256;
    occupancy_request.registers_per_thread = 32;
    occupancy_request.max_blocks_sm = 24;
    const auto thread_limited = common_tree_draft_sm89_occupancy(sm89, occupancy_request);
    assert(thread_limited.available);
    assert(thread_limited.resident_blocks == 6);
    assert(thread_limited.resident_warps == 48);
    assert(thread_limited.occupancy == 1.0f);
    assert(thread_limited.limiting_resource == COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_THREADS);

    occupancy_request.registers_per_thread = 64;
    const auto register_limited = common_tree_draft_sm89_occupancy(sm89, occupancy_request);
    assert(register_limited.available);
    assert(register_limited.resident_blocks == 4);
    assert(register_limited.resident_warps == 32);
    assert(register_limited.limiting_resource == COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_REGISTERS);

    occupancy_request.registers_per_thread = 0;
    occupancy_request.static_smem_bytes = 32768;
    occupancy_request.dynamic_smem_bytes = 4096;
    const auto smem_limited = common_tree_draft_sm89_occupancy(sm89, occupancy_request);
    assert(smem_limited.available);
    assert(smem_limited.resident_blocks == 2);
    assert(smem_limited.resident_warps == 16);
    assert(smem_limited.limiting_resource == COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_SHARED_MEMORY);

    occupancy_request.static_smem_bytes = 0;
    occupancy_request.dynamic_smem_bytes = 0;
    occupancy_request.max_blocks_sm = 2;
    const auto block_limited = common_tree_draft_sm89_occupancy(sm89, occupancy_request);
    assert(block_limited.available);
    assert(block_limited.resident_blocks == 2);
    assert(block_limited.limiting_resource == COMMON_TREE_DRAFT_OCCUPANCY_LIMIT_BLOCKS);

    occupancy_request.block_size = 2048;
    assert(!common_tree_draft_sm89_occupancy(sm89, occupancy_request).available);

    common_tree_draft_vector_access_request vector_request = {};
    vector_request.address = 0x1000;
    vector_request.stride_bytes = 64;
    vector_request.element_count = 32;
    vector_request.element_size = 2;
    assert(common_tree_draft_vector_access_is_aligned(vector_request, 16));
    const auto vector16 = common_tree_draft_sm89_vector_access(sm89, vector_request);
    assert(vector16.available);
    assert(vector16.vectorized);
    assert(vector16.vector_bytes == 16);
    assert(vector16.elements_per_vector == 8);

    vector_request.address += 8;
    const auto vector8 = common_tree_draft_sm89_vector_access(sm89, vector_request);
    assert(vector8.available);
    assert(vector8.vectorized);
    assert(vector8.vector_bytes == 8);
    assert(vector8.elements_per_vector == 4);

    vector_request.address += 4;
    const auto vector4 = common_tree_draft_sm89_vector_access(sm89, vector_request);
    assert(vector4.available);
    assert(vector4.vectorized);
    assert(vector4.vector_bytes == 4);
    assert(vector4.elements_per_vector == 2);

    vector_request.address += 2;
    const auto scalar_misaligned = common_tree_draft_sm89_vector_access(sm89, vector_request);
    assert(scalar_misaligned.available);
    assert(!scalar_misaligned.vectorized);
    assert(scalar_misaligned.vector_bytes == 2);
    assert(scalar_misaligned.elements_per_vector == 1);

    vector_request.address = 0x1000;
    vector_request.element_count = 10;
    const auto count_limited = common_tree_draft_sm89_vector_access(sm89, vector_request);
    assert(count_limited.available);
    assert(count_limited.vectorized);
    assert(count_limited.vector_bytes == 4);
    assert(count_limited.elements_per_vector == 2);

    vector_request.element_count = 9;
    const auto scalar_tail = common_tree_draft_sm89_vector_access(sm89, vector_request);
    assert(scalar_tail.available);
    assert(!scalar_tail.vectorized);
    assert(scalar_tail.vector_bytes == 2);

    vector_request.element_count = 32;
    vector_request.stride_bytes = 12;
    const auto stride_limited = common_tree_draft_sm89_vector_access(sm89, vector_request);
    assert(stride_limited.available);
    assert(stride_limited.vectorized);
    assert(stride_limited.vector_bytes == 4);

    vector_request.element_size = 3;
    const auto odd_element = common_tree_draft_sm89_vector_access(sm89, vector_request);
    assert(odd_element.available);
    assert(!odd_element.vectorized);
    assert(odd_element.vector_bytes == 3);

    vector_request.element_size = 0;
    assert(!common_tree_draft_sm89_vector_access(sm89, vector_request).available);

    auto raw_sm80 = raw_sm89;
    raw_sm80.cc_minor = 0;
    const auto sm80 = common_tree_draft_device_capabilities_normalize(raw_sm80);
    assert(!common_tree_draft_device_is_sm89(sm80));
    assert(!common_tree_draft_sm89_launch_geometry_candidates(sm80).available);

    vector_request.element_size = 2;
    vector_request.element_count = 32;
    vector_request.stride_bytes = 64;
    const auto scalar_non_sm89 = common_tree_draft_sm89_vector_access(sm80, vector_request);
    assert(scalar_non_sm89.available);
    assert(!scalar_non_sm89.vectorized);
    assert(scalar_non_sm89.vector_bytes == 2);

    auto oversized = raw_sm89;
    oversized.cc_major = 300;
    oversized.warp_size = 100000;
    oversized.async_engines = 1000;
    const auto saturated = common_tree_draft_device_capabilities_normalize(oversized);
    assert(saturated.cc_major == std::numeric_limits<uint8_t>::max());
    assert(saturated.warp_size == std::numeric_limits<uint16_t>::max());
    assert(saturated.async_engines == std::numeric_limits<uint8_t>::max());

    common_tree_draft_device_capability_cache cache(fake_sm89_probe);
    ggml_backend_dev_t fake_device_0 = reinterpret_cast<ggml_backend_dev_t>((uintptr_t) 1);
    ggml_backend_dev_t fake_device_1 = reinterpret_cast<ggml_backend_dev_t>((uintptr_t) 2);
    assert(common_tree_draft_device_is_sm89(cache.get(fake_device_0)));
    assert(common_tree_draft_device_is_sm89(cache.get(fake_device_0)));
    assert(g_probe_calls == 1);
    assert(common_tree_draft_device_is_sm89(cache.get(fake_device_1)));
    assert(g_probe_calls == 2);

    common_tree_draft_device_capability_cache failed_cache(fake_failed_probe);
    const auto failed_0 = failed_cache.get(fake_device_0);
    const auto failed_1 = failed_cache.get(fake_device_0);
    assert(!failed_0.available);
    assert(!failed_1.available);
    assert(g_failed_probe_calls == 1);

    const auto null_fallback = common_tree_draft_device_capabilities_get(nullptr);
    assert(!null_fallback.available);
    assert(!common_tree_draft_device_is_sm89(null_fallback));

    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu) {
        const auto cpu_fallback = common_tree_draft_device_capabilities_get(cpu);
        assert(!cpu_fallback.available);
        assert(!common_tree_draft_device_is_sm89(cpu_fallback));
    }

    return 0;
}
