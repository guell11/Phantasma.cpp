#include "tree-draft-visibility-decode.h"
#include "tree-draft-ancestor.h"

#include <cassert>
#include <vector>

int main() {
    const int32_t parent[]  = {-1,0,0,1,1,2};
    const int32_t depth[]   = { 0,1,1,2,2,2};
    const int32_t tree_id[] = { 5,5,5,5,5,5};
    std::vector<uint32_t> words(6, 0);
    assert(common_tree_draft_ancestor_build({parent,depth,tree_id,6},{words.data(),words.size()}) == COMMON_TREE_DRAFT_ANCESTOR_OK);
    common_tree_draft_ancestor_format format{words.data(),6,1,COMMON_TREE_DRAFT_ANCESTOR_WORD_32};

    assert(common_tree_draft_visibility_decode(format, 3, 0));
    assert(common_tree_draft_visibility_decode(format, 3, 1));
    assert(common_tree_draft_visibility_decode(format, 3, 3));
    assert(!common_tree_draft_visibility_decode(format, 3, 2));
    assert(!common_tree_draft_visibility_decode(format, 3, 6));
    assert(!common_tree_draft_visibility_decode(format, 99, 0));

    common_tree_draft_visibility_lane lanes[] = {{0,9},{1,9},{2,9},{3,9},{5,9},{6,9},{999,9}};
    common_tree_draft_visibility_decode_lanes(format,3,lanes,7);
    const uint8_t expected[] = {1,1,0,1,0,0,0};
    for (size_t i = 0; i < 7; ++i) assert(lanes[i].visible == expected[i]);
    return 0;
}
