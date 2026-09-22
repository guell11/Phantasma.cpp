#include "tree-draft-external-quant.h"

#include <algorithm>
#include <cmath>
#include <limits>

static const std::string * meta(const common_tree_draft_safetensors_table & table, const char * key) {
    const auto it = table.metadata.find(key);
    return it == table.metadata.end() ? nullptr : &it->second;
}

static bool parse_u32(const std::string & text, uint32_t * value) {
    try {
        size_t pos = 0;
        const unsigned long long v = std::stoull(text, &pos, 10);
        if (pos != text.size() || v > UINT32_MAX) return false;
        *value = static_cast<uint32_t>(v);
        return true;
    } catch (...) { return false; }
}

static bool parse_i32(const std::string & text, int32_t * value) {
    try {
        size_t pos = 0;
        const long long v = std::stoll(text, &pos, 10);
        if (pos != text.size() || v < INT32_MIN || v > INT32_MAX) return false;
        *value = static_cast<int32_t>(v);
        return true;
    } catch (...) { return false; }
}

static bool parse_bool(const std::string & text, bool * value) {
    if (text == "true" || text == "1") { *value = true; return true; }
    if (text == "false" || text == "0") { *value = false; return true; }
    return false;
}

static bool parse_double(const std::string & text, double * value) {
    try {
        size_t pos = 0;
        const double v = std::stod(text, &pos);
        if (pos != text.size() || !std::isfinite(v)) return false;
        *value = v;
        return true;
    } catch (...) { return false; }
}

static const common_tree_draft_safetensors_tensor * tensor(const common_tree_draft_safetensors_table & table, const std::string & name) {
    for (const auto & t : table.tensors) if (t.name == name) return &t;
    return nullptr;
}

static uint64_t elements(const common_tree_draft_safetensors_tensor & t, bool * ok) {
    uint64_t n = 1;
    *ok = !t.shape.empty();
    for (uint64_t d : t.shape) {
        if (d == 0 || n > UINT64_MAX / d) { *ok = false; return 0; }
        n *= d;
    }
    return n;
}

common_tree_draft_external_quant_status common_tree_draft_gptq_parse(
        const common_tree_draft_safetensors_table & table,
        common_tree_draft_gptq_descriptor * descriptor) {
    if (descriptor == nullptr) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED;
    const std::string * method = meta(table, "quant_method");
    if (method == nullptr || *method != "gptq") return COMMON_TREE_DRAFT_EXTERNAL_QUANT_NOT_PRESENT;
    const std::string * bits = meta(table, "bits");
    const std::string * group = meta(table, "group_size");
    const std::string * sym = meta(table, "sym");
    const std::string * desc = meta(table, "desc_act");
    if (!bits || !group || !sym || !desc) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED;
    common_tree_draft_gptq_descriptor out;
    if (!parse_u32(*bits, &out.bits) || (out.bits != 2 && out.bits != 3 && out.bits != 4 && out.bits != 8) ||
        !parse_i32(*group, &out.group_size) || (out.group_size <= 0 && out.group_size != -1) ||
        !parse_bool(*sym, &out.sym) || !parse_bool(*desc, &out.desc_act)) {
        return COMMON_TREE_DRAFT_EXTERNAL_QUANT_UNSUPPORTED;
    }
    if (const auto * damp = meta(table, "damp_percent")) {
        double v = 0.0;
        if (!parse_double(*damp, &v) || v < 0.0 || v > 1.0) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED;
        out.damp_percent = v;
    }
    if (const auto * packing = meta(table, "packing_version")) out.packing_version = *packing;
    *descriptor = std::move(out);
    return COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK;
}

common_tree_draft_external_quant_status common_tree_draft_awq_parse(
        const common_tree_draft_safetensors_table & table,
        common_tree_draft_awq_descriptor * descriptor) {
    if (descriptor == nullptr) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED;
    const std::string * method = meta(table, "quant_method");
    if (method == nullptr || *method != "awq") return COMMON_TREE_DRAFT_EXTERNAL_QUANT_NOT_PRESENT;
    const std::string * bits = meta(table, "bits");
    const std::string * group = meta(table, "group_size");
    const std::string * zero = meta(table, "zero_point");
    if (!bits || !group || !zero) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED;
    common_tree_draft_awq_descriptor out;
    if (!parse_u32(*bits, &out.bits) || (out.bits != 4 && out.bits != 8) ||
        !parse_u32(*group, &out.group_size) || out.group_size == 0 || !parse_bool(*zero, &out.zero_point)) {
        return COMMON_TREE_DRAFT_EXTERNAL_QUANT_UNSUPPORTED;
    }
    if (const auto * packing = meta(table, "packing_version")) out.packing_version = *packing;
    *descriptor = std::move(out);
    return COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK;
}

common_tree_draft_external_quant_status common_tree_draft_bnb8_parse(
        const common_tree_draft_safetensors_table & table,
        const std::string & prefix,
        common_tree_draft_bnb8_descriptor * descriptor) {
    if (descriptor == nullptr || prefix.empty()) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED;
    const auto * weight = tensor(table, prefix + ".weight");
    const auto * scale = tensor(table, prefix + ".SCB");
    if (weight == nullptr && scale == nullptr) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_NOT_PRESENT;
    if (weight == nullptr || scale == nullptr) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MISSING_TENSOR;
    if (weight->dtype != "I8" || weight->shape.size() != 2 || scale->shape.size() != 1 || scale->shape[0] != weight->shape[0]) {
        return COMMON_TREE_DRAFT_EXTERNAL_QUANT_SHAPE;
    }
    common_tree_draft_bnb8_descriptor out;
    out.weight_tensor = weight->name;
    out.scale_tensor = scale->name;
    if (tensor(table, prefix + ".outlier_map")) out.outlier_tensor = prefix + ".outlier_map";
    out.rows = weight->shape[0];
    out.cols = weight->shape[1];
    *descriptor = std::move(out);
    return COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK;
}

common_tree_draft_external_quant_status common_tree_draft_bnb4_parse(
        const common_tree_draft_safetensors_table & table,
        const std::string & prefix,
        common_tree_draft_bnb4_descriptor * descriptor) {
    if (descriptor == nullptr || prefix.empty()) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED;
    const auto * weight = tensor(table, prefix + ".weight");
    const auto * absmax = tensor(table, prefix + ".absmax");
    const std::string * kind = meta(table, "bnb_4bit_quant_type");
    const std::string * block = meta(table, "bnb_4bit_block_size");
    if (weight == nullptr && absmax == nullptr && kind == nullptr) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_NOT_PRESENT;
    if (!weight || !absmax || !kind || !block) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MISSING_TENSOR;
    common_tree_draft_bnb4_descriptor out;
    if (*kind == "nf4") out.kind = COMMON_TREE_DRAFT_BNB4_NF4;
    else if (*kind == "fp4") out.kind = COMMON_TREE_DRAFT_BNB4_FP4;
    else return COMMON_TREE_DRAFT_EXTERNAL_QUANT_UNSUPPORTED;
    if (!parse_u32(*block, &out.block_size) || out.block_size == 0) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MALFORMED;
    bool ok = false;
    out.logical_elements = elements(*weight, &ok);
    if (!ok) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_SHAPE;
    const uint64_t blocks = (out.logical_elements + out.block_size - 1) / out.block_size;
    bool scale_ok = false;
    const uint64_t scale_elements = elements(*absmax, &scale_ok);
    if (!scale_ok || scale_elements != blocks) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_SHAPE;
    out.weight_tensor = weight->name;
    out.absmax_tensor = absmax->name;
    out.nested_depth = 0;
    if (const auto * nested = meta(table, "bnb_4bit_nested_depth")) {
        if (!parse_u32(*nested, &out.nested_depth) || out.nested_depth > 2) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_UNSUPPORTED;
        if (out.nested_depth > 0) {
            const std::string nested_name = prefix + ".nested_absmax";
            if (tensor(table, nested_name) == nullptr) return COMMON_TREE_DRAFT_EXTERNAL_QUANT_MISSING_TENSOR;
            out.nested_absmax_tensor = nested_name;
        }
    }
    *descriptor = std::move(out);
    return COMMON_TREE_DRAFT_EXTERNAL_QUANT_OK;
}

bool common_tree_draft_gptq_equal(const common_tree_draft_gptq_descriptor & a, const common_tree_draft_gptq_descriptor & b) {
    return a.bits == b.bits && a.group_size == b.group_size && a.sym == b.sym && a.desc_act == b.desc_act &&
           a.damp_percent == b.damp_percent && a.packing_version == b.packing_version;
}

bool common_tree_draft_awq_equal(const common_tree_draft_awq_descriptor & a, const common_tree_draft_awq_descriptor & b) {
    return a.bits == b.bits && a.group_size == b.group_size && a.zero_point == b.zero_point && a.packing_version == b.packing_version;
}

