#include "tree-draft-alignment.h"

#include <cassert>
#include <limits>

int main() {
    common_tree_draft_alignment_table table = {};
    assert(common_tree_draft_alignment_build({16, 256, 128, 64}, &table) == COMMON_TREE_DRAFT_ALIGNMENT_OK);
    assert(table.required[COMMON_TREE_DRAFT_BUFFER_HOST_STAGING] == 256);
    assert(table.required[COMMON_TREE_DRAFT_BUFFER_DEVICE_SCRATCH] == 128);
    assert(table.required[COMMON_TREE_DRAFT_BUFFER_KV_PAGE] == 256);
    assert(table.required[COMMON_TREE_DRAFT_BUFFER_TENSOR_TILE] == 128);
    assert(table.required[COMMON_TREE_DRAFT_BUFFER_METADATA] == 64);
    assert(common_tree_draft_alignment_offset_valid(table, COMMON_TREE_DRAFT_BUFFER_KV_PAGE, 512));
    assert(!common_tree_draft_alignment_offset_valid(table, COMMON_TREE_DRAFT_BUFFER_KV_PAGE, 384));

    uint64_t aligned = 0;
    assert(common_tree_draft_alignment_up(257, 256, &aligned) == COMMON_TREE_DRAFT_ALIGNMENT_OK && aligned == 512);
    assert(common_tree_draft_alignment_up(std::numeric_limits<uint64_t>::max(), 256, &aligned) == COMMON_TREE_DRAFT_ALIGNMENT_OVERFLOW);
    assert(common_tree_draft_alignment_build({16, 24, 128, 64}, &table) == COMMON_TREE_DRAFT_ALIGNMENT_INVALID);
    return 0;
}

