#include "tree-draft-host.h"

#include <cassert>
#include <type_traits>

static void test_request_and_result_envelopes_preserve_host_data() {
    static_assert(std::is_standard_layout<common_tree_draft_token_span>::value, "token span must support C adapters");
    static_assert(std::is_trivially_copyable<common_tree_draft_token_span>::value, "token span must have value semantics");

    const int32_t input[] = { 1, 2, 3 };
    common_tree_draft_request request = {
        /* .request_id = */ "request-1",
        /* .session_id = */ "session-1",
        /* .input_tokens = */ { input, 3 },
        /* .sampling = */ { 0.7f, 0.9f, 40, 99 },
        /* .limits = */ { 32, 64, 8, 16 },
        /* .metadata = */ { { "user", "local" } },
    };

    common_tree_draft_result result = {
        /* .request_id = */ request.request_id,
        /* .status = */ COMMON_TREE_DRAFT_RESULT_STATUS_OK,
        /* .accepted_tokens = */ { 1, 2 },
        /* .output_tokens = */ { 1, 2, 4 },
        /* .usage = */ { 3, 3, 2 },
        /* .finish_reason = */ COMMON_TREE_DRAFT_FINISH_REASON_STOP,
    };

    assert(result.request_id == request.request_id);
    assert(request.input_tokens.data == input);
    assert(request.input_tokens.size == 3);
    assert(request.sampling.temperature == 0.7f);
    assert(request.sampling.top_p == 0.9f);
    assert(request.sampling.top_k == 40);
    assert(request.sampling.seed == 99);
    assert(request.limits.max_tokens == 32);
    assert(request.metadata.size() == 1);
    assert(request.metadata[0].key == "user");
    assert(request.metadata[0].value == "local");
    assert(result.accepted_tokens.size() == 2);
    assert(result.output_tokens.size() == 3);
    assert(result.usage.accepted_tokens == result.accepted_tokens.size());
    assert(result.finish_reason == COMMON_TREE_DRAFT_FINISH_REASON_STOP);
}

int main() {
    test_request_and_result_envelopes_preserve_host_data();
    return 0;
}
