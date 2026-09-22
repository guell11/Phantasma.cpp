#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

enum ggml_backend_cuda_tensor_core_mode {
    GGML_BACKEND_CUDA_TENSOR_CORE_F16  = 1u << 0,
    GGML_BACKEND_CUDA_TENSOR_CORE_BF16 = 1u << 1,
    GGML_BACKEND_CUDA_TENSOR_CORE_TF32 = 1u << 2,
    GGML_BACKEND_CUDA_TENSOR_CORE_FP8  = 1u << 3,
};

struct ggml_backend_cuda_device_capabilities {
    uint32_t cc_major;
    uint32_t cc_minor;
    uint32_t warp_size;
    uint32_t max_threads_sm;
    uint32_t max_threads_block;
    uint32_t regs_sm;
    uint64_t smem_sm;
    uint64_t l2_bytes;
    uint32_t async_engines;
    uint32_t tensor_core_modes;
};

typedef bool (*ggml_backend_cuda_get_device_capabilities_t)(
        ggml_backend_dev_t device,
        struct ggml_backend_cuda_device_capabilities * capabilities);

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
