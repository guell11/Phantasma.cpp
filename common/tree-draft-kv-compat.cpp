#include "tree-draft-kv-compat.h"

common_tree_draft_kv_compatibility common_tree_draft_kv_compatibility_report(
        ggml_type type_k,
        ggml_type type_v,
        const common_tree_draft_kv_capabilities & capabilities) {
    const bool quantized_k = ggml_is_quantized(type_k);
    const bool quantized_v = ggml_is_quantized(type_v);

    const bool direct_compatible = capabilities.direct_k && capabilities.direct_v;
    const bool existing_backend_compatible = capabilities.existing_backend_exact;
    const bool staging_compatible = capabilities.staging_exact;

    common_tree_draft_kv_path selected_path = COMMON_TREE_DRAFT_KV_PATH_UNSUPPORTED;
    if (direct_compatible) {
        selected_path = COMMON_TREE_DRAFT_KV_PATH_DIRECT;
    } else if (existing_backend_compatible) {
        selected_path = COMMON_TREE_DRAFT_KV_PATH_EXISTING_BACKEND;
    } else if (staging_compatible) {
        selected_path = COMMON_TREE_DRAFT_KV_PATH_STAGING;
    }

    return {
        type_k,
        type_v,
        quantized_k,
        quantized_v,
        direct_compatible,
        existing_backend_compatible,
        staging_compatible,
        selected_path,
    };
}

const char * common_tree_draft_kv_path_name(common_tree_draft_kv_path path) {
    switch (path) {
        case COMMON_TREE_DRAFT_KV_PATH_DIRECT:           return "direct";
        case COMMON_TREE_DRAFT_KV_PATH_EXISTING_BACKEND: return "existing_backend";
        case COMMON_TREE_DRAFT_KV_PATH_STAGING:          return "staging";
        case COMMON_TREE_DRAFT_KV_PATH_UNSUPPORTED:      return "unsupported";
    }
    return "unknown";
}
