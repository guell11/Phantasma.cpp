#include "tree-draft-ancestor-format.h"

#include <cstdint>

uint32_t common_tree_draft_ancestor_format_min_words_per_row(
        uint32_t n_nodes,
        common_tree_draft_ancestor_word_bits word_bits) {
    const uint32_t bits = static_cast<uint32_t>(word_bits);
    if (bits != 32 && bits != 64) return 0;
    return n_nodes == 0 ? 0 : (n_nodes + bits - 1) / bits;
}

common_tree_draft_ancestor_format_status common_tree_draft_ancestor_format_validate(
        const common_tree_draft_ancestor_format & format) {
    const uint32_t bits = static_cast<uint32_t>(format.word_bits);
    if (bits != 32 && bits != 64) return COMMON_TREE_DRAFT_ANCESTOR_FORMAT_WORD_BITS;
    const uint32_t min_words = common_tree_draft_ancestor_format_min_words_per_row(format.n_nodes, format.word_bits);
    if (format.n_nodes == 0) {
        return format.words_per_row == 0 ? COMMON_TREE_DRAFT_ANCESTOR_FORMAT_OK : COMMON_TREE_DRAFT_ANCESTOR_FORMAT_ROW_STRIDE;
    }
    if (format.words == nullptr) return COMMON_TREE_DRAFT_ANCESTOR_FORMAT_NULL;
    if (format.words_per_row < min_words) return COMMON_TREE_DRAFT_ANCESTOR_FORMAT_ROW_STRIDE;
    const uintptr_t address = reinterpret_cast<uintptr_t>(format.words);
    const uintptr_t alignment = bits == 64 ? alignof(uint64_t) : alignof(uint32_t);
    if (address % alignment != 0) return COMMON_TREE_DRAFT_ANCESTOR_FORMAT_ALIGNMENT;
    return COMMON_TREE_DRAFT_ANCESTOR_FORMAT_OK;
}

bool common_tree_draft_ancestor_format_contains(
        const common_tree_draft_ancestor_format & format,
        uint32_t node,
        uint32_t candidate) {
    if (common_tree_draft_ancestor_format_validate(format) != COMMON_TREE_DRAFT_ANCESTOR_FORMAT_OK ||
        node >= format.n_nodes || candidate >= format.n_nodes) return false;
    const uint32_t bits = static_cast<uint32_t>(format.word_bits);
    const size_t index = static_cast<size_t>(node) * format.words_per_row + candidate / bits;
    const uint32_t shift = candidate % bits;
    if (bits == 64) {
        const auto * words = static_cast<const uint64_t *>(format.words);
        return ((words[index] >> shift) & uint64_t{1}) != 0;
    }
    const auto * words = static_cast<const uint32_t *>(format.words);
    return ((words[index] >> shift) & uint32_t{1}) != 0;
}
