#include "tree-draft-error.h"

#include <cassert>
#include <cstring>

struct expected_error_mapping {
    common_tree_draft_engine_error engine_error;
    common_tree_draft_error_domain domain;
    common_tree_draft_error_code code;
    bool retryable;
};

static void test_engine_error_mapping_is_stable() {
    const expected_error_mapping mappings[] = {
        { COMMON_TREE_DRAFT_ENGINE_ERROR_INVALID_ARGUMENT,    COMMON_TREE_DRAFT_ERROR_DOMAIN_INVALID_ARGUMENT,   COMMON_TREE_DRAFT_ERROR_CODE_INVALID_ARGUMENT,    false },
        { COMMON_TREE_DRAFT_ENGINE_ERROR_RESOURCE_EXHAUSTED,  COMMON_TREE_DRAFT_ERROR_DOMAIN_RESOURCE_EXHAUSTED, COMMON_TREE_DRAFT_ERROR_CODE_RESOURCE_EXHAUSTED,  true  },
        { COMMON_TREE_DRAFT_ENGINE_ERROR_CANCELLED,           COMMON_TREE_DRAFT_ERROR_DOMAIN_CANCELLED,          COMMON_TREE_DRAFT_ERROR_CODE_CANCELLED,           false },
        { COMMON_TREE_DRAFT_ENGINE_ERROR_DEADLINE_EXCEEDED,   COMMON_TREE_DRAFT_ERROR_DOMAIN_DEADLINE,           COMMON_TREE_DRAFT_ERROR_CODE_DEADLINE_EXCEEDED,   true  },
        { COMMON_TREE_DRAFT_ENGINE_ERROR_UNAVAILABLE,         COMMON_TREE_DRAFT_ERROR_DOMAIN_UNAVAILABLE,        COMMON_TREE_DRAFT_ERROR_CODE_UNAVAILABLE,         true  },
        { COMMON_TREE_DRAFT_ENGINE_ERROR_INTERNAL,            COMMON_TREE_DRAFT_ERROR_DOMAIN_INTERNAL,           COMMON_TREE_DRAFT_ERROR_CODE_INTERNAL,            false },
        { COMMON_TREE_DRAFT_ENGINE_ERROR_PROTOCOL,            COMMON_TREE_DRAFT_ERROR_DOMAIN_PROTOCOL,           COMMON_TREE_DRAFT_ERROR_CODE_PROTOCOL,            false },
    };

    for (const auto & mapping : mappings) {
        const auto first = common_tree_draft_error_from_engine_error(mapping.engine_error, "first");
        const auto second = common_tree_draft_error_from_engine_error(mapping.engine_error, "second");
        assert(first.domain == mapping.domain);
        assert(first.code == mapping.code);
        assert(first.retryable == mapping.retryable);
        assert(second.domain == first.domain);
        assert(second.code == first.code);
    }
}

static void test_error_preserves_payload_fields() {
    const auto error = common_tree_draft_error_from_engine_error(
            COMMON_TREE_DRAFT_ENGINE_ERROR_RESOURCE_EXHAUSTED,
            "cache full",
            "request-1",
            { { "limit", "context" } });

    assert(error.message == "cache full");
    assert(error.request_id == "request-1");
    assert(error.details.size() == 1);
    assert(error.details[0].key == "limit");
    assert(error.details[0].value == "context");
}

static void test_unknown_engine_errors_map_to_internal() {
    const auto error = common_tree_draft_error_from_engine_error((common_tree_draft_engine_error) -1);
    assert(error.domain == COMMON_TREE_DRAFT_ERROR_DOMAIN_INTERNAL);
    assert(error.code == COMMON_TREE_DRAFT_ERROR_CODE_INTERNAL);
    assert(!error.retryable);
}

static void test_stable_names() {
    assert(std::strcmp(common_tree_draft_error_domain_name(COMMON_TREE_DRAFT_ERROR_DOMAIN_RESOURCE_EXHAUSTED), "resource_exhausted") == 0);
    assert(std::strcmp(common_tree_draft_error_code_name(COMMON_TREE_DRAFT_ERROR_CODE_DEADLINE_EXCEEDED), "deadline_exceeded") == 0);
}

int main() {
    test_engine_error_mapping_is_stable();
    test_error_preserves_payload_fields();
    test_unknown_engine_errors_map_to_internal();
    test_stable_names();
    return 0;
}
