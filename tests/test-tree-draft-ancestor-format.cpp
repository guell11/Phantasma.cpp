#include "tree-draft-ancestor-format.h"
#include "tree-draft-ancestor.h"

#include <cassert>
#include <cstdint>
#include <vector>

int main() {
    assert(common_tree_draft_ancestor_format_min_words_per_row(65, COMMON_TREE_DRAFT_ANCESTOR_WORD_32) == 3);
    assert(common_tree_draft_ancestor_format_min_words_per_row(65, COMMON_TREE_DRAFT_ANCESTOR_WORD_64) == 2);

    const int32_t parent[]  = {-1,0,0,1};
    const int32_t depth[]   = { 0,1,1,2};
    const int32_t tree_id[] = { 7,7,7,7};
    std::vector<uint32_t> words(4, 0);
    assert(common_tree_draft_ancestor_build({parent,depth,tree_id,4},{words.data(),words.size()}) == COMMON_TREE_DRAFT_ANCESTOR_OK);
    common_tree_draft_ancestor_format f32{words.data(),4,1,COMMON_TREE_DRAFT_ANCESTOR_WORD_32};
    assert(common_tree_draft_ancestor_format_validate(f32) == COMMON_TREE_DRAFT_ANCESTOR_FORMAT_OK);
    assert(common_tree_draft_ancestor_format_contains(f32,3,0));
    assert(common_tree_draft_ancestor_format_contains(f32,3,1));
    assert(common_tree_draft_ancestor_format_contains(f32,3,3));
    assert(!common_tree_draft_ancestor_format_contains(f32,3,2));

    // Same logical numbering with uint64 words, including the 63/64 boundary.
    std::vector<uint64_t> w64(4, 0);
    w64[0] = uint64_t{1} << 63;
    w64[1] = uint64_t{1};
    common_tree_draft_ancestor_format f64{w64.data(),65,2,COMMON_TREE_DRAFT_ANCESTOR_WORD_64};
    assert(common_tree_draft_ancestor_format_validate(f64) == COMMON_TREE_DRAFT_ANCESTOR_FORMAT_OK);
    assert(common_tree_draft_ancestor_format_contains(f64,0,63));
    assert(common_tree_draft_ancestor_format_contains(f64,0,64));
    assert(!common_tree_draft_ancestor_format_contains(f64,0,62));

    auto short_row = f64; short_row.words_per_row = 1;
    assert(common_tree_draft_ancestor_format_validate(short_row) == COMMON_TREE_DRAFT_ANCESTOR_FORMAT_ROW_STRIDE);
    common_tree_draft_ancestor_format empty{nullptr,0,0,COMMON_TREE_DRAFT_ANCESTOR_WORD_32};
    assert(common_tree_draft_ancestor_format_validate(empty) == COMMON_TREE_DRAFT_ANCESTOR_FORMAT_OK);
    return 0;
}
