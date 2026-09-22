#pragma once

#include "tree-draft-quant-registry.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_dequant_output_dtype : uint32_t {
    COMMON_TREE_DRAFT_DEQUANT_F32 = 0,
};

using common_tree_draft_dequant_adapter_fn = bool (*)(
        const void * encoded,
        size_t encoded_bytes,
        uint64_t logical_elements,
        common_tree_draft_dequant_output_dtype output_dtype,
        void * output,
        size_t output_bytes,
        void * user_data);

struct common_tree_draft_dequant_adapter {
    common_tree_draft_quant_source source = COMMON_TREE_DRAFT_QUANT_GGML;
    common_tree_draft_dequant_adapter_fn fn = nullptr;
    void * user_data = nullptr;
};

enum common_tree_draft_dequant_status : uint32_t {
    COMMON_TREE_DRAFT_DEQUANT_OK = 0,
    COMMON_TREE_DRAFT_DEQUANT_NULL_BUFFER,
    COMMON_TREE_DRAFT_DEQUANT_UNSUPPORTED,
    COMMON_TREE_DRAFT_DEQUANT_RANGE,
    COMMON_TREE_DRAFT_DEQUANT_SIZE_MISMATCH,
    COMMON_TREE_DRAFT_DEQUANT_FAILED,
};

common_tree_draft_dequant_status common_tree_draft_dequant_cpu(
        const common_tree_draft_quant_type & quant,
        const void * encoded,
        size_t encoded_bytes,
        uint64_t logical_elements,
        common_tree_draft_dequant_output_dtype output_dtype,
        void * output,
        size_t output_bytes,
        const common_tree_draft_dequant_adapter * adapters = nullptr,
        size_t adapter_count = 0);

double common_tree_draft_dequant_max_abs_error(const float * actual, const float * reference, size_t count);
double common_tree_draft_dequant_default_tolerance(const common_tree_draft_quant_type & quant);

