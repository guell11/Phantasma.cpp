#include "tree-draft-top-k.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>

static void assert_sentinel(const common_tree_draft_top_k_entry & entry) {
    assert(entry.token == COMMON_TREE_DRAFT_TOP_K_SENTINEL_TOKEN);
    assert(std::isinf(entry.logit));
    assert(entry.logit < 0.0f);
}

int main() {
    const float logits[] = { 1.0f, 3.0f, 3.0f, -2.0f };

    {
        common_tree_draft_top_k_entry out[2];
        size_t count = 99;
        assert(common_tree_draft_top_k_reference(logits, 4, 0, out, 2, &count) == COMMON_TREE_DRAFT_TOP_K_OK);
        assert(count == 0);
        assert_sentinel(out[0]);
        assert_sentinel(out[1]);
    }

    {
        common_tree_draft_top_k_entry out[1];
        size_t count = 0;
        assert(common_tree_draft_top_k_reference(logits, 4, 1, out, 1, &count) == COMMON_TREE_DRAFT_TOP_K_OK);
        assert(count == 1);
        assert(out[0].token == 1);
        assert(out[0].logit == 3.0f);
    }

    {
        common_tree_draft_top_k_entry out[6];
        size_t count = 0;
        assert(common_tree_draft_top_k_reference(logits, 4, 6, out, 6, &count) == COMMON_TREE_DRAFT_TOP_K_OK);
        assert(count == 4);
        const int32_t expected[] = { 1, 2, 0, 3 };
        for (size_t i = 0; i < count; ++i) {
            assert(out[i].token == expected[i]);
        }
        assert_sentinel(out[4]);
        assert_sentinel(out[5]);
    }

    {
        const float tied[] = { 4.0f, 4.0f, 4.0f, 4.0f };
        common_tree_draft_top_k_entry out[4];
        size_t count = 0;
        assert(common_tree_draft_top_k_reference(tied, 4, 4, out, 4, &count) == COMMON_TREE_DRAFT_TOP_K_OK);
        assert(count == 4);
        for (int32_t token = 0; token < 4; ++token) {
            assert(out[token].token == token);
        }
    }

    {
        const float non_finite[] = {
            -std::numeric_limits<float>::infinity(),
            2.0f,
            std::numeric_limits<float>::infinity(),
            -0.0f,
        };
        common_tree_draft_top_k_entry out[4];
        size_t count = 0;
        assert(common_tree_draft_top_k_reference(non_finite, 4, 4, out, 4, &count) == COMMON_TREE_DRAFT_TOP_K_OK);
        assert(count == 4);
        assert(out[0].token == 2 && std::isinf(out[0].logit) && out[0].logit > 0.0f);
        assert(out[1].token == 1 && out[1].logit == 2.0f);
        assert(out[2].token == 3 && std::signbit(out[2].logit));
        assert(out[3].token == 0 && std::isinf(out[3].logit) && out[3].logit < 0.0f);
    }

    {
        const float with_nan[] = { 5.0f, std::numeric_limits<float>::quiet_NaN(), 4.0f };
        common_tree_draft_top_k_entry out[3] = { { 7, 7.0f }, { 8, 8.0f }, { 9, 9.0f } };
        size_t count = 99;
        assert(common_tree_draft_top_k_reference(with_nan, 3, 2, out, 3, &count) == COMMON_TREE_DRAFT_TOP_K_NAN_LOGIT);
        assert(count == 0);
        assert_sentinel(out[0]);
        assert_sentinel(out[1]);
        assert_sentinel(out[2]);

        count = 99;
        assert(common_tree_draft_top_k_reference(with_nan, 3, 0, out, 3, &count) == COMMON_TREE_DRAFT_TOP_K_NAN_LOGIT);
        assert(count == 0);
    }

    {
        common_tree_draft_top_k_entry out[1];
        size_t count = 99;
        assert(common_tree_draft_top_k_reference(logits, 4, 2, out, 1, &count) == COMMON_TREE_DRAFT_TOP_K_BUFFER_TOO_SMALL);
        assert(count == 0);
        assert_sentinel(out[0]);
        assert(std::strcmp(common_tree_draft_top_k_error_name(COMMON_TREE_DRAFT_TOP_K_BUFFER_TOO_SMALL), "buffer_too_small") == 0);
    }

    return 0;
}
