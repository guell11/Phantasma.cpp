#include "tree-draft-repack-plan.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static void hash_bytes(uint64_t & h, const void * data, size_t size) {
    const auto * p = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) h = mix64(h ^ p[i]);
}

static void hash_string(uint64_t & h, const std::string & s) {
    hash_bytes(h, s.data(), s.size());
    h = mix64(h ^ s.size());
}

static void hash_layout(uint64_t & h, const common_tree_draft_tensor_layout & l) {
    h = mix64(h ^ static_cast<uint64_t>(l.kind));
    auto hash_vec = [&](const auto & v) {
        for (const auto & x : v) h = mix64(h ^ static_cast<uint64_t>(x));
        h = mix64(h ^ v.size());
    };
    hash_vec(l.logical_axes);
    hash_vec(l.storage_axes);
    hash_vec(l.permutation);
    hash_vec(l.storage_shape);
    hash_vec(l.logical_shape);
    hash_vec(l.block_shape);
    hash_vec(l.strides);
    h = mix64(h ^ static_cast<uint64_t>(l.allow_tail_block));
}

common_tree_draft_repack_status common_tree_draft_repack_plan_build(
        const common_tree_draft_repack_request & request,
        common_tree_draft_repack_plan * plan) {
    if (plan == nullptr || request.quant == nullptr || request.source.source_id.empty() ||
        request.source.source_hash.empty() || request.target_kernel.empty() ||
        request.kernel_abi_version == 0 || !std::isfinite(request.tolerance) || request.tolerance < 0.0) {
        return COMMON_TREE_DRAFT_REPACK_INVALID;
    }
    if (request.source.tensor_bytes == 0 ||
        request.source.tensor_offset > std::numeric_limits<uint64_t>::max() - request.source.tensor_bytes) {
        return COMMON_TREE_DRAFT_REPACK_SOURCE_RANGE;
    }
    if (common_tree_draft_tensor_layout_validate(request.source_layout) != COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK ||
        common_tree_draft_tensor_layout_validate(request.target_layout) != COMMON_TREE_DRAFT_TENSOR_LAYOUT_OK) {
        return COMMON_TREE_DRAFT_REPACK_LAYOUT;
    }

    uint64_t h = UINT64_C(0x7068616e7461736d);
    hash_string(h, request.source.source_id);
    hash_string(h, request.source.source_hash);
    h = mix64(h ^ request.source.tensor_offset);
    h = mix64(h ^ request.source.tensor_bytes);
    h = mix64(h ^ request.quant->id);
    hash_layout(h, request.source_layout);
    hash_layout(h, request.target_layout);
    hash_string(h, request.target_kernel);
    h = mix64(h ^ request.kernel_abi_version);

    std::ostringstream key;
    key << "td-repack-v1:" << std::hex << std::setfill('0') << std::setw(16) << h;

    common_tree_draft_repack_plan out;
    out.cache_key = key.str();
    out.source = request.source;
    out.quant_id = request.quant->id;
    out.source_layout = request.source_layout.kind;
    out.target_layout = request.target_layout.kind;
    out.target_kernel = request.target_kernel;
    out.kernel_abi_version = request.kernel_abi_version;
    out.tolerance = request.tolerance;
    *plan = std::move(out);
    return COMMON_TREE_DRAFT_REPACK_OK;
}

bool common_tree_draft_repack_equivalent(
        const float * source_decoded,
        const float * repacked_decoded,
        size_t count,
        double tolerance,
        double * max_abs_error) {
    if ((count > 0 && (source_decoded == nullptr || repacked_decoded == nullptr)) ||
        !std::isfinite(tolerance) || tolerance < 0.0) {
        if (max_abs_error) *max_abs_error = std::numeric_limits<double>::infinity();
        return false;
    }
    double max_err = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double a = source_decoded[i];
        const double b = repacked_decoded[i];
        if (!std::isfinite(a) || !std::isfinite(b)) {
            if (max_abs_error) *max_abs_error = std::numeric_limits<double>::infinity();
            return false;
        }
        max_err = std::max(max_err, std::fabs(a - b));
    }
    if (max_abs_error) *max_abs_error = max_err;
    return max_err <= tolerance;
}
