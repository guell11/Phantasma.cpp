#include "tree-draft-kernel-capability.h"

static bool valid_entry(const common_tree_draft_kernel_capability_entry & e) {
    return e.device_cc != 0 && e.alignment_bytes != 0 &&
           (e.alignment_bytes & (e.alignment_bytes - 1)) == 0 &&
           e.k_multiple != 0 && e.n_multiple != 0;
}

common_tree_draft_kernel_capability_status common_tree_draft_kernel_capability_lookup(
        const common_tree_draft_device_capabilities & device,
        const common_tree_draft_kernel_capability_entry * entries,
        size_t entry_count,
        const common_tree_draft_kernel_capability_query & query,
        common_tree_draft_kernel_capability_result * result) {
    if (result == nullptr || query.quant == nullptr) {
        return COMMON_TREE_DRAFT_KERNEL_CAPABILITY_INVALID_QUERY;
    }
    if (entry_count > 0 && entries == nullptr) {
        return COMMON_TREE_DRAFT_KERNEL_CAPABILITY_INVALID_REGISTRY;
    }
    for (size_t i = 0; i < entry_count; ++i) {
        if (!valid_entry(entries[i])) return COMMON_TREE_DRAFT_KERNEL_CAPABILITY_INVALID_REGISTRY;
    }

    common_tree_draft_kernel_capability_result out;
    const uint32_t cc = static_cast<uint32_t>(device.cc_major) * 10u + device.cc_minor;
    for (size_t i = 0; i < entry_count; ++i) {
        const auto & e = entries[i];
        if (!device.available || e.device_cc != cc ||
            e.quant_source != query.quant->source || e.quant_id != query.quant->id ||
            e.op != query.op || e.layout != query.layout ||
            e.compute_dtype != query.compute_dtype || e.group_size != query.group_size) {
            continue;
        }
        if (query.address % e.alignment_bytes != 0 ||
            query.k % e.k_multiple != 0 || query.n % e.n_multiple != 0 ||
            (query.has_tail_group && !e.supports_tail_group)) {
            continue;
        }
        out.supported = true;
        out.entry_index = static_cast<uint32_t>(i);
        out.required_alignment = e.alignment_bytes;
        out.required_k_multiple = e.k_multiple;
        out.required_n_multiple = e.n_multiple;
        out.fused_dequant = e.fused_dequant;
        break;
    }
    *result = out;
    return COMMON_TREE_DRAFT_KERNEL_CAPABILITY_OK;
}
