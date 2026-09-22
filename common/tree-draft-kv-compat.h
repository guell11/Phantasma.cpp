#pragma once

#include "tree-draft-kv-indirection.h"

#include "ggml.h"

#include <cstdint>

enum common_tree_draft_kv_path : int32_t {
    COMMON_TREE_DRAFT_KV_PATH_DIRECT = 0,
    COMMON_TREE_DRAFT_KV_PATH_EXISTING_BACKEND,
    COMMON_TREE_DRAFT_KV_PATH_STAGING,
    COMMON_TREE_DRAFT_KV_PATH_UNSUPPORTED,
};

struct common_tree_draft_kv_capabilities {
    bool direct_k;
    bool direct_v;
    bool existing_backend_exact;
    bool staging_exact;
};

struct common_tree_draft_kv_compatibility {
    ggml_type type_k;
    ggml_type type_v;
    bool quantized_k;
    bool quantized_v;
    bool direct_compatible;
    bool existing_backend_compatible;
    bool staging_compatible;
    common_tree_draft_kv_path selected_path;
};

common_tree_draft_kv_compatibility common_tree_draft_kv_compatibility_report(
        ggml_type type_k,
        ggml_type type_v,
        const common_tree_draft_kv_capabilities & capabilities);

const char * common_tree_draft_kv_path_name(common_tree_draft_kv_path path);
