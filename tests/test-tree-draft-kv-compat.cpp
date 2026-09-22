#include "tree-draft-kv-compat.h"

#include <cassert>
#include <string>

static void assert_report(
        ggml_type type_k,
        ggml_type type_v,
        common_tree_draft_kv_capabilities capabilities,
        common_tree_draft_kv_path expected_path,
        bool expected_quantized_k,
        bool expected_quantized_v) {
    const auto report = common_tree_draft_kv_compatibility_report(type_k, type_v, capabilities);
    assert(report.type_k == type_k);
    assert(report.type_v == type_v);
    assert(report.quantized_k == expected_quantized_k);
    assert(report.quantized_v == expected_quantized_v);
    assert(report.selected_path == expected_path);
}

int main() {
    assert_report(
        GGML_TYPE_F16,
        GGML_TYPE_F16,
        { true, true, true, true },
        COMMON_TREE_DRAFT_KV_PATH_DIRECT,
        false,
        false);

    assert_report(
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q4_0,
        { false, false, true, true },
        COMMON_TREE_DRAFT_KV_PATH_EXISTING_BACKEND,
        true,
        true);

    assert_report(
        GGML_TYPE_Q8_0,
        GGML_TYPE_F16,
        { false, true, false, true },
        COMMON_TREE_DRAFT_KV_PATH_STAGING,
        true,
        false);

    assert_report(
        GGML_TYPE_F16,
        GGML_TYPE_Q5_1,
        { true, false, false, false },
        COMMON_TREE_DRAFT_KV_PATH_UNSUPPORTED,
        false,
        true);

    {
        const auto report = common_tree_draft_kv_compatibility_report(
            GGML_TYPE_Q4_1,
            GGML_TYPE_Q5_0,
            { true, false, true, true });
        assert(!report.direct_compatible);
        assert(report.existing_backend_compatible);
        assert(report.staging_compatible);
        assert(report.selected_path == COMMON_TREE_DRAFT_KV_PATH_EXISTING_BACKEND);
    }

    {
        const auto report = common_tree_draft_kv_compatibility_report(
            GGML_TYPE_BF16,
            GGML_TYPE_Q4_0,
            { true, false, false, true });
        assert(!report.direct_compatible);
        assert(!report.existing_backend_compatible);
        assert(report.staging_compatible);
        assert(report.selected_path == COMMON_TREE_DRAFT_KV_PATH_STAGING);
    }

    assert(std::string(common_tree_draft_kv_path_name(COMMON_TREE_DRAFT_KV_PATH_DIRECT)) == "direct");
    assert(std::string(common_tree_draft_kv_path_name(COMMON_TREE_DRAFT_KV_PATH_EXISTING_BACKEND)) == "existing_backend");
    assert(std::string(common_tree_draft_kv_path_name(COMMON_TREE_DRAFT_KV_PATH_STAGING)) == "staging");
    assert(std::string(common_tree_draft_kv_path_name(COMMON_TREE_DRAFT_KV_PATH_UNSUPPORTED)) == "unsupported");

    return 0;
}
