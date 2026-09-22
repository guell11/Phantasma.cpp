#include "tree-draft-dequant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

common_tree_draft_dequant_status common_tree_draft_dequant_cpu(
        const common_tree_draft_quant_type & quant,
        const void * encoded,
        size_t encoded_bytes,
        uint64_t logical_elements,
        common_tree_draft_dequant_output_dtype output_dtype,
        void * output,
        size_t output_bytes,
        const common_tree_draft_dequant_adapter * adapters,
        size_t adapter_count) {
    if (encoded == nullptr || output == nullptr) return COMMON_TREE_DRAFT_DEQUANT_NULL_BUFFER;
    if (logical_elements == 0 || logical_elements > static_cast<uint64_t>(INT64_MAX)) return COMMON_TREE_DRAFT_DEQUANT_RANGE;
    if (output_dtype != COMMON_TREE_DRAFT_DEQUANT_F32) return COMMON_TREE_DRAFT_DEQUANT_UNSUPPORTED;
    if (logical_elements > SIZE_MAX / sizeof(float) || output_bytes != static_cast<size_t>(logical_elements) * sizeof(float)) {
        return COMMON_TREE_DRAFT_DEQUANT_SIZE_MISMATCH;
    }

    if (quant.source == COMMON_TREE_DRAFT_QUANT_GGML && quant.ggml_type_id >= 0) {
        if (!quant.fixed_block_size || quant.block_elems == 0 || quant.block_bytes == 0 || logical_elements % quant.block_elems != 0) {
            return COMMON_TREE_DRAFT_DEQUANT_RANGE;
        }
        const uint64_t blocks = logical_elements / quant.block_elems;
        if (blocks > SIZE_MAX / quant.block_bytes || encoded_bytes != static_cast<size_t>(blocks * quant.block_bytes)) {
            return COMMON_TREE_DRAFT_DEQUANT_SIZE_MISMATCH;
        }
        const auto type = static_cast<enum ggml_type>(quant.ggml_type_id);
        if (type == GGML_TYPE_F32) {
            if (encoded_bytes != output_bytes) return COMMON_TREE_DRAFT_DEQUANT_SIZE_MISMATCH;
            std::memcpy(output, encoded, encoded_bytes);
            return COMMON_TREE_DRAFT_DEQUANT_OK;
        }
        const ggml_type_traits * traits = ggml_get_type_traits(type);
        if (traits == nullptr || traits->to_float == nullptr) return COMMON_TREE_DRAFT_DEQUANT_UNSUPPORTED;
        traits->to_float(encoded, static_cast<float *>(output), static_cast<int64_t>(logical_elements));
        return COMMON_TREE_DRAFT_DEQUANT_OK;
    }

    for (size_t i = 0; i < adapter_count; ++i) {
        if (adapters[i].source == quant.source && adapters[i].fn != nullptr) {
            return adapters[i].fn(encoded, encoded_bytes, logical_elements, output_dtype, output, output_bytes, adapters[i].user_data)
                ? COMMON_TREE_DRAFT_DEQUANT_OK : COMMON_TREE_DRAFT_DEQUANT_FAILED;
        }
    }
    return COMMON_TREE_DRAFT_DEQUANT_UNSUPPORTED;
}

double common_tree_draft_dequant_max_abs_error(const float * actual, const float * reference, size_t count) {
    if (count > 0 && (actual == nullptr || reference == nullptr)) return std::numeric_limits<double>::infinity();
    double max_error = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double error = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(reference[i]));
        if (!std::isfinite(error)) return std::numeric_limits<double>::infinity();
        max_error = std::max(max_error, error);
    }
    return max_error;
}

double common_tree_draft_dequant_default_tolerance(const common_tree_draft_quant_type & quant) {
    if (quant.source != COMMON_TREE_DRAFT_QUANT_GGML || quant.ggml_type_id < 0) return 0.0;
    switch (static_cast<enum ggml_type>(quant.ggml_type_id)) {
        case GGML_TYPE_F32: return 0.0;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: return 1e-2;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q8_K: return 5e-2;
        default: return 0.5;
    }
}

