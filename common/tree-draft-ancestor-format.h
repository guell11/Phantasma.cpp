#pragma once

#include <cstddef>
#include <cstdint>

enum common_tree_draft_ancestor_word_bits : uint32_t {
    COMMON_TREE_DRAFT_ANCESTOR_WORD_32 = 32,
    COMMON_TREE_DRAFT_ANCESTOR_WORD_64 = 64,
};

// Portable row-major ancestor bitset ABI shared with GPU/Triton-facing code.
// Logical node j is always LSB-first: word=j/W, bit=j%W.
struct common_tree_draft_ancestor_format {
    const void * words = nullptr;
    uint32_t n_nodes = 0;
    uint32_t words_per_row = 0;
    common_tree_draft_ancestor_word_bits word_bits = COMMON_TREE_DRAFT_ANCESTOR_WORD_32;
};

enum common_tree_draft_ancestor_format_status : uint32_t {
    COMMON_TREE_DRAFT_ANCESTOR_FORMAT_OK = 0,
    COMMON_TREE_DRAFT_ANCESTOR_FORMAT_NULL,
    COMMON_TREE_DRAFT_ANCESTOR_FORMAT_WORD_BITS,
    COMMON_TREE_DRAFT_ANCESTOR_FORMAT_ROW_STRIDE,
    COMMON_TREE_DRAFT_ANCESTOR_FORMAT_ALIGNMENT,
};

uint32_t common_tree_draft_ancestor_format_min_words_per_row(
        uint32_t n_nodes,
        common_tree_draft_ancestor_word_bits word_bits);

common_tree_draft_ancestor_format_status common_tree_draft_ancestor_format_validate(
        const common_tree_draft_ancestor_format & format);

bool common_tree_draft_ancestor_format_contains(
        const common_tree_draft_ancestor_format & format,
        uint32_t node,
        uint32_t candidate);
