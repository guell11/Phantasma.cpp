#include "tree-draft-packed-linear.h"

static const common_tree_draft_safetensors_tensor * find_tensor(const common_tree_draft_safetensors_table & table, const std::string & name) {
    for (const auto & tensor : table.tensors) if (tensor.name == name) return &tensor;
    return nullptr;
}

static bool packed_int_dtype(const std::string & dtype) {
    return dtype == "I32" || dtype == "U32" || dtype == "I8" || dtype == "U8";
}

static common_tree_draft_packed_linear_status bind_common(
        const common_tree_draft_safetensors_table & table,
        const std::string & prefix,
        common_tree_draft_packed_linear_format format,
        uint32_t bits,
        uint32_t group_size,
        bool symmetric,
        bool zero_point,
        bool allow_g_idx,
        uint64_t out_features,
        uint64_t in_features,
        common_tree_draft_packed_linear_view * view) {
    if (!view || prefix.empty() || out_features == 0 || in_features == 0) return COMMON_TREE_DRAFT_PACKED_LINEAR_SHAPE;
    if (bits == 0 || bits > 32 || 32 % bits != 0) return COMMON_TREE_DRAFT_PACKED_LINEAR_BITS;
    if (group_size == 0 || (group_size != UINT32_MAX && group_size > in_features)) return COMMON_TREE_DRAFT_PACKED_LINEAR_GROUP;
    const uint32_t unpack = 32 / bits;
    const uint64_t effective_group = group_size == UINT32_MAX ? in_features : group_size;
    const uint64_t groups = (in_features + effective_group - 1) / effective_group;
    const auto * qweight = find_tensor(table, prefix + ".qweight");
    const auto * qzeros = find_tensor(table, prefix + ".qzeros");
    const auto * scales = find_tensor(table, prefix + ".scales");
    if (!qweight || !qzeros || !scales) return COMMON_TREE_DRAFT_PACKED_LINEAR_MISSING;
    if (!packed_int_dtype(qweight->dtype) || !packed_int_dtype(qzeros->dtype) ||
        (scales->dtype != "F16" && scales->dtype != "BF16" && scales->dtype != "F32")) return COMMON_TREE_DRAFT_PACKED_LINEAR_DTYPE;
    if (qweight->shape.size() != 2 || qzeros->shape.size() != 2 || scales->shape.size() != 2) return COMMON_TREE_DRAFT_PACKED_LINEAR_SHAPE;

    const uint64_t packed_in = (in_features + unpack - 1) / unpack;
    const uint64_t packed_out = (out_features + unpack - 1) / unpack;
    const bool qweight_normal = qweight->shape[0] == packed_in && qweight->shape[1] == out_features;
    const bool qweight_transposed = qweight->shape[0] == out_features && qweight->shape[1] == packed_in;
    if (!qweight_normal && !qweight_transposed) return COMMON_TREE_DRAFT_PACKED_LINEAR_SHAPE;
    const bool scales_ok = (scales->shape[0] == groups && scales->shape[1] == out_features) ||
                           (scales->shape[0] == out_features && scales->shape[1] == groups);
    const bool zeros_ok = (qzeros->shape[0] == groups && qzeros->shape[1] == packed_out) ||
                          (qzeros->shape[0] == packed_out && qzeros->shape[1] == groups);
    if (!scales_ok || !zeros_ok) return COMMON_TREE_DRAFT_PACKED_LINEAR_SHAPE;

    common_tree_draft_packed_linear_view out;
    out.format = format;
    out.qweight = qweight->name;
    out.qzeros = qzeros->name;
    out.scales = scales->name;
    if (allow_g_idx && find_tensor(table, prefix + ".g_idx")) out.g_idx = prefix + ".g_idx";
    out.out_features = out_features;
    out.in_features = in_features;
    out.bits = bits;
    out.unpack_factor = unpack;
    out.group_size = static_cast<uint32_t>(effective_group);
    out.group_count = groups;
    out.symmetric = symmetric;
    out.zero_point = zero_point;
    out.storage_transposed = qweight_transposed;
    *view = std::move(out);
    return COMMON_TREE_DRAFT_PACKED_LINEAR_OK;
}

common_tree_draft_packed_linear_status common_tree_draft_gptq_bind(
        const common_tree_draft_safetensors_table & table, const std::string & prefix,
        const common_tree_draft_gptq_descriptor & quant, uint64_t out_features, uint64_t in_features,
        common_tree_draft_packed_linear_view * view) {
    const uint32_t group = quant.group_size == -1 ? UINT32_MAX : static_cast<uint32_t>(quant.group_size);
    return bind_common(table, prefix, COMMON_TREE_DRAFT_PACKED_LINEAR_GPTQ, quant.bits, group,
            quant.sym, !quant.sym, true, out_features, in_features, view);
}

common_tree_draft_packed_linear_status common_tree_draft_awq_bind(
        const common_tree_draft_safetensors_table & table, const std::string & prefix,
        const common_tree_draft_awq_descriptor & quant, uint64_t out_features, uint64_t in_features,
        common_tree_draft_packed_linear_view * view) {
    return bind_common(table, prefix, COMMON_TREE_DRAFT_PACKED_LINEAR_AWQ, quant.bits, quant.group_size,
            false, quant.zero_point, false, out_features, in_features, view);
}

