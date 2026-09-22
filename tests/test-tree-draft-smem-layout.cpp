#include "tree-draft-smem-layout.h"

#include <cassert>

int main() {
    const common_tree_draft_smem_component c[] = {
        {COMMON_TREE_DRAFT_SMEM_LOGITS_TILE, 1000, 16},
        {COMMON_TREE_DRAFT_SMEM_TOKEN_IDS, 120, 8},
        {COMMON_TREE_DRAFT_SMEM_PREFIX_META, 96, 32},
        {COMMON_TREE_DRAFT_SMEM_REDUCTION_SCRATCH, 512, 16},
    };
    common_tree_draft_smem_layout out{};
    assert(common_tree_draft_smem_layout_build({256,4096,c,4},&out) == COMMON_TREE_DRAFT_SMEM_LAYOUT_OK);
    assert(out.components[0].offset == 0);
    assert(out.components[1].offset == 1000);
    assert(out.components[2].offset == 1120);
    assert(out.components[3].offset == 1216);
    assert(out.dynamic_shared_bytes == 1728);
    assert(out.total_shared_bytes == 1984);

    auto bad = c[0]; bad.alignment = 3;
    assert(common_tree_draft_smem_layout_build({0,4096,&bad,1},&out) == COMMON_TREE_DRAFT_SMEM_LAYOUT_BAD_ALIGNMENT);
    assert(common_tree_draft_smem_layout_build({3000,3500,c,4},&out) == COMMON_TREE_DRAFT_SMEM_LAYOUT_EXCEEDS_LIMIT);
    return 0;
}
