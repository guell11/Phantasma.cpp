#include "tree-draft-kv-identity.h"

#include <cassert>
#include <cstdint>
#include <cstring>

static common_tree_draft_kv_token_identity token(
        int32_t node,
        int64_t position,
        uint32_t branch,
        uint32_t page,
        uint32_t generation,
        uint32_t offset) {
    return {
        node,
        position,
        { branch, 1 },
        { { page, generation }, offset },
        -1,
        true,
    };
}

int main() {
    const int32_t parent[]  = { -1, 0, 0, 1 };
    const int32_t depth[]   = {  0, 1, 1, 2 };
    const int32_t tree_id[] = {  7, 7, 7, 7 };
    const common_tree_draft_topology topology = { parent, depth, tree_id, 4 };

    common_tree_draft_kv_token_identity tokens[] = {
        token(0, 0, 10, 3, 4, 0),
        token(1, 1, 10, 3, 4, 1),
        token(2, 1, 11, 3, 4, 2),
        token(3, 2, 10, 8, 2, 0),
    };
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_OK);

    const auto saved = tokens[3];
    tokens[3].logical_node = 2;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_LOGICAL_NODE_MISMATCH);
    tokens[3] = saved;

    tokens[3].logical_position = 3;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_POSITION_MISMATCH);
    tokens[3] = saved;

    tokens[3].address = tokens[1].address;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_DUPLICATE_ADDRESS);
    tokens[3].shared_from_node = 1;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_OK);
    tokens[3].shared_from_node = 2;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_SHARED_ANCESTOR);
    tokens[3] = saved;

    tokens[1].address = tokens[0].address;
    tokens[1].shared_from_node = 0;
    tokens[3].address = tokens[1].address;
    tokens[3].shared_from_node = 1;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_OK);
    tokens[1] = token(1, 1, 10, 3, 4, 1);
    tokens[3] = saved;

    tokens[3].materialized = false;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_UNMATERIALIZED_ADDRESS);
    tokens[3].address = { { COMMON_TREE_DRAFT_KV_ID_INVALID, 0 }, COMMON_TREE_DRAFT_KV_OFFSET_INVALID };
    tokens[3].shared_from_node = -1;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_OK);
    tokens[3] = saved;

    tokens[3].branch.id = COMMON_TREE_DRAFT_KV_ID_INVALID;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_BRANCH);
    tokens[3] = saved;

    tokens[3].address.page.id = COMMON_TREE_DRAFT_KV_ID_INVALID;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_PAGE);
    tokens[3] = saved;

    tokens[3].address.offset = COMMON_TREE_DRAFT_KV_OFFSET_INVALID;
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_OFFSET);
    tokens[3] = saved;

    assert(common_tree_draft_kv_identity_validate(topology, nullptr, 4) == COMMON_TREE_DRAFT_KV_IDENTITY_NULL_TOKENS);
    assert(common_tree_draft_kv_identity_validate(topology, tokens, 3) == COMMON_TREE_DRAFT_KV_IDENTITY_TOKEN_COUNT_MISMATCH);

    const common_tree_draft_branch_handle branch_a = { 5, 9 };
    const common_tree_draft_branch_handle branch_b = { 5, 9 };
    const common_tree_draft_branch_handle branch_stale = { 5, 8 };
    assert(common_tree_draft_branch_handle_equal(branch_a, branch_b));
    assert(!common_tree_draft_branch_handle_equal(branch_a, branch_stale));

    const common_tree_draft_page_address page_a = { { 2, 3 }, 7 };
    const common_tree_draft_page_address page_b = { { 2, 3 }, 7 };
    const common_tree_draft_page_address page_reused = { { 2, 4 }, 7 };
    assert(common_tree_draft_page_address_equal(page_a, page_b));
    assert(!common_tree_draft_page_address_equal(page_a, page_reused));
    assert(std::strcmp(common_tree_draft_kv_identity_error_name(COMMON_TREE_DRAFT_KV_IDENTITY_DUPLICATE_ADDRESS), "duplicate_address") == 0);

    const common_tree_draft_topology empty = { nullptr, nullptr, nullptr, 0 };
    assert(common_tree_draft_kv_identity_validate(empty, nullptr, 0) == COMMON_TREE_DRAFT_KV_IDENTITY_OK);
    return 0;
}
