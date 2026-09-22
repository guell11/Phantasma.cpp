#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum common_tree_draft_loader_error_class : uint32_t {
    COMMON_TREE_DRAFT_LOADER_CORRUPT = 0,
    COMMON_TREE_DRAFT_LOADER_UNSUPPORTED,
    COMMON_TREE_DRAFT_LOADER_RESOURCE,
    COMMON_TREE_DRAFT_LOADER_POLICY,
    COMMON_TREE_DRAFT_LOADER_INTERNAL,
};

struct common_tree_draft_loader_error {
    common_tree_draft_loader_error_class category = COMMON_TREE_DRAFT_LOADER_INTERNAL;
    std::string stage;
    std::string source;
    std::string message;
    bool declared_recoverable = false;
};

struct common_tree_draft_loader_attempt {
    std::string strategy;
    common_tree_draft_loader_error error;
};

enum common_tree_draft_loader_fallback_decision : uint32_t {
    COMMON_TREE_DRAFT_LOADER_FALLBACK_TERMINAL = 0,
    COMMON_TREE_DRAFT_LOADER_FALLBACK_ALLOWED,
};

common_tree_draft_loader_fallback_decision common_tree_draft_loader_fallback(
        const common_tree_draft_loader_error & error);

bool common_tree_draft_loader_attempt_chain_valid(
        const std::vector<common_tree_draft_loader_attempt> & attempts);

const char * common_tree_draft_loader_error_class_name(common_tree_draft_loader_error_class category);

