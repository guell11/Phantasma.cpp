#include "tree-draft-loader-error.h"

common_tree_draft_loader_fallback_decision common_tree_draft_loader_fallback(
        const common_tree_draft_loader_error & error) {
    if (!error.declared_recoverable) return COMMON_TREE_DRAFT_LOADER_FALLBACK_TERMINAL;
    switch (error.category) {
        case COMMON_TREE_DRAFT_LOADER_UNSUPPORTED:
        case COMMON_TREE_DRAFT_LOADER_RESOURCE:
        case COMMON_TREE_DRAFT_LOADER_POLICY:
            return COMMON_TREE_DRAFT_LOADER_FALLBACK_ALLOWED;
        case COMMON_TREE_DRAFT_LOADER_CORRUPT:
        case COMMON_TREE_DRAFT_LOADER_INTERNAL:
            return COMMON_TREE_DRAFT_LOADER_FALLBACK_TERMINAL;
    }
    return COMMON_TREE_DRAFT_LOADER_FALLBACK_TERMINAL;
}

bool common_tree_draft_loader_attempt_chain_valid(
        const std::vector<common_tree_draft_loader_attempt> & attempts) {
    for (size_t i = 0; i < attempts.size(); ++i) {
        const auto & attempt = attempts[i];
        if (attempt.strategy.empty() || attempt.error.stage.empty() || attempt.error.message.empty()) return false;
        if (i + 1 < attempts.size() && common_tree_draft_loader_fallback(attempt.error) != COMMON_TREE_DRAFT_LOADER_FALLBACK_ALLOWED) {
            return false;
        }
    }
    return true;
}

const char * common_tree_draft_loader_error_class_name(common_tree_draft_loader_error_class category) {
    switch (category) {
        case COMMON_TREE_DRAFT_LOADER_CORRUPT:     return "corrupt";
        case COMMON_TREE_DRAFT_LOADER_UNSUPPORTED: return "unsupported";
        case COMMON_TREE_DRAFT_LOADER_RESOURCE:    return "resource";
        case COMMON_TREE_DRAFT_LOADER_POLICY:      return "policy";
        case COMMON_TREE_DRAFT_LOADER_INTERNAL:    return "internal";
    }
    return "internal";
}

