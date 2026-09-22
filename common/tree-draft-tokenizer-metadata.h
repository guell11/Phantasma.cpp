#pragma once

#include "tree-draft-model-metadata.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

enum common_tree_draft_tokenizer_compatibility {
    COMMON_TREE_DRAFT_TOKENIZER_EXACT = 0,
    COMMON_TREE_DRAFT_TOKENIZER_PARTIAL,
    COMMON_TREE_DRAFT_TOKENIZER_INCOMPATIBLE,
};

struct common_tree_draft_special_token {
    std::string role;
    uint64_t id = 0;
};

struct common_tree_draft_tokenizer_signature {
    std::optional<std::string> tokenizer_family;
    std::optional<uint64_t> vocab_size;
    std::vector<common_tree_draft_special_token> special_tokens;
    bool has_special_token_metadata = false;
    bool special_token_metadata_valid = true;
    std::optional<std::string> vocab_hash;
    std::optional<std::string> merge_or_model_hash;
};

struct common_tree_draft_tokenizer_compatibility_result {
    common_tree_draft_tokenizer_compatibility status = COMMON_TREE_DRAFT_TOKENIZER_PARTIAL;
    std::vector<std::string> mismatched_fields;
    std::vector<std::string> missing_fields;
};

common_tree_draft_tokenizer_signature common_tree_draft_tokenizer_metadata_normalize(
        const common_tree_draft_model_metadata & metadata);

common_tree_draft_tokenizer_compatibility_result common_tree_draft_tokenizer_compare(
        const common_tree_draft_tokenizer_signature & lhs,
        const common_tree_draft_tokenizer_signature & rhs);
