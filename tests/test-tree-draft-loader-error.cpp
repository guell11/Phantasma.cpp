#include "tree-draft-loader-error.h"

#include <cassert>

int main() {
    common_tree_draft_loader_error unsupported = {
        COMMON_TREE_DRAFT_LOADER_UNSUPPORTED, "mmap", "model.gguf", "mapping unavailable", true
    };
    assert(common_tree_draft_loader_fallback(unsupported) == COMMON_TREE_DRAFT_LOADER_FALLBACK_ALLOWED);
    common_tree_draft_loader_error resource = {
        COMMON_TREE_DRAFT_LOADER_RESOURCE, "placement", "model.gguf", "device memory budget", true
    };
    assert(common_tree_draft_loader_fallback(resource) == COMMON_TREE_DRAFT_LOADER_FALLBACK_ALLOWED);
    common_tree_draft_loader_error corrupt = {
        COMMON_TREE_DRAFT_LOADER_CORRUPT, "integrity", "model.gguf", "digest mismatch", true
    };
    assert(common_tree_draft_loader_fallback(corrupt) == COMMON_TREE_DRAFT_LOADER_FALLBACK_TERMINAL);
    common_tree_draft_loader_error internal = {
        COMMON_TREE_DRAFT_LOADER_INTERNAL, "planner", "model.gguf", "invariant", true
    };
    assert(common_tree_draft_loader_fallback(internal) == COMMON_TREE_DRAFT_LOADER_FALLBACK_TERMINAL);

    std::vector<common_tree_draft_loader_attempt> good = {
        {"mmap", unsupported},
        {"buffered", {COMMON_TREE_DRAFT_LOADER_POLICY, "policy", "model.gguf", "user forces cpu", true}},
        {"cpu", {COMMON_TREE_DRAFT_LOADER_UNSUPPORTED, "kernel", "model.gguf", "no native kernel", false}},
    };
    assert(common_tree_draft_loader_attempt_chain_valid(good));
    std::vector<common_tree_draft_loader_attempt> bad = {
        {"mmap", corrupt},
        {"buffered", unsupported},
    };
    assert(!common_tree_draft_loader_attempt_chain_valid(bad));
    return 0;
}

