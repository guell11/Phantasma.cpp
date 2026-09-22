#include "tree-draft-kv-identity.h"

bool common_tree_draft_branch_handle_equal(
        const common_tree_draft_branch_handle & a,
        const common_tree_draft_branch_handle & b) {
    return a.id == b.id && a.generation == b.generation;
}

bool common_tree_draft_page_handle_equal(
        const common_tree_draft_page_handle & a,
        const common_tree_draft_page_handle & b) {
    return a.id == b.id && a.generation == b.generation;
}

bool common_tree_draft_page_address_equal(
        const common_tree_draft_page_address & a,
        const common_tree_draft_page_address & b) {
    return common_tree_draft_page_handle_equal(a.page, b.page) && a.offset == b.offset;
}

static bool common_tree_draft_kv_is_ancestor(
        const common_tree_draft_topology & topology,
        int32_t ancestor,
        int32_t node) {
    int32_t current = node;
    while (current >= 0) {
        if (current == ancestor) {
            return true;
        }
        current = topology.parent[current];
    }
    return false;
}

common_tree_draft_kv_identity_error common_tree_draft_kv_identity_validate(
        const common_tree_draft_topology & topology,
        const common_tree_draft_kv_token_identity * tokens,
        size_t token_count) {
    if (common_tree_draft_topology_validate(topology) != COMMON_TREE_DRAFT_TOPOLOGY_OK) {
        return COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_TOPOLOGY;
    }
    if (token_count != static_cast<size_t>(topology.n_nodes)) {
        return COMMON_TREE_DRAFT_KV_IDENTITY_TOKEN_COUNT_MISMATCH;
    }
    if (token_count > 0 && tokens == nullptr) {
        return COMMON_TREE_DRAFT_KV_IDENTITY_NULL_TOKENS;
    }

    for (int32_t i = 0; i < topology.n_nodes; ++i) {
        const common_tree_draft_kv_token_identity & token = tokens[i];
        if (token.logical_node != i) {
            return COMMON_TREE_DRAFT_KV_IDENTITY_LOGICAL_NODE_MISMATCH;
        }
        if (token.logical_position != topology.depth[i]) {
            return COMMON_TREE_DRAFT_KV_IDENTITY_POSITION_MISMATCH;
        }
        if (token.branch.id == COMMON_TREE_DRAFT_KV_ID_INVALID) {
            return COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_BRANCH;
        }

        if (!token.materialized) {
            if (token.address.page.id != COMMON_TREE_DRAFT_KV_ID_INVALID ||
                    token.address.offset != COMMON_TREE_DRAFT_KV_OFFSET_INVALID ||
                    token.shared_from_node != -1) {
                return COMMON_TREE_DRAFT_KV_IDENTITY_UNMATERIALIZED_ADDRESS;
            }
            continue;
        }
        if (token.address.page.id == COMMON_TREE_DRAFT_KV_ID_INVALID) {
            return COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_PAGE;
        }
        if (token.address.offset == COMMON_TREE_DRAFT_KV_OFFSET_INVALID) {
            return COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_OFFSET;
        }
        if (token.shared_from_node < -1 || token.shared_from_node >= i) {
            return COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_SHARED_ANCESTOR;
        }
        if (token.shared_from_node >= 0 &&
                (!common_tree_draft_kv_is_ancestor(topology, token.shared_from_node, i) ||
                 !tokens[token.shared_from_node].materialized ||
                 !common_tree_draft_page_address_equal(token.address, tokens[token.shared_from_node].address))) {
            return COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_SHARED_ANCESTOR;
        }
        bool duplicate_address = false;
        for (int32_t j = 0; j < i; ++j) {
            duplicate_address = duplicate_address ||
                (tokens[j].materialized && common_tree_draft_page_address_equal(token.address, tokens[j].address));
        }
        if (duplicate_address && token.shared_from_node < 0) {
            return COMMON_TREE_DRAFT_KV_IDENTITY_DUPLICATE_ADDRESS;
        }
    }

    return COMMON_TREE_DRAFT_KV_IDENTITY_OK;
}

const char * common_tree_draft_kv_identity_error_name(common_tree_draft_kv_identity_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_KV_IDENTITY_OK:                     return "ok";
        case COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_TOPOLOGY:       return "invalid_topology";
        case COMMON_TREE_DRAFT_KV_IDENTITY_NULL_TOKENS:            return "null_tokens";
        case COMMON_TREE_DRAFT_KV_IDENTITY_TOKEN_COUNT_MISMATCH:   return "token_count_mismatch";
        case COMMON_TREE_DRAFT_KV_IDENTITY_LOGICAL_NODE_MISMATCH:  return "logical_node_mismatch";
        case COMMON_TREE_DRAFT_KV_IDENTITY_POSITION_MISMATCH:      return "position_mismatch";
        case COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_BRANCH:         return "invalid_branch";
        case COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_PAGE:           return "invalid_page";
        case COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_OFFSET:         return "invalid_offset";
        case COMMON_TREE_DRAFT_KV_IDENTITY_UNMATERIALIZED_ADDRESS: return "unmaterialized_address";
        case COMMON_TREE_DRAFT_KV_IDENTITY_INVALID_SHARED_ANCESTOR:return "invalid_shared_ancestor";
        case COMMON_TREE_DRAFT_KV_IDENTITY_DUPLICATE_ADDRESS:      return "duplicate_address";
    }
    return "unknown";
}
