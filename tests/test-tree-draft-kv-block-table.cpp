#include "tree-draft-kv-block-table.h"

#include <cassert>

int main() {
    uint32_t row[]={0,2,3};
    common_tree_draft_kv_gather_segment seg[]={
        {{7,2},0,4},{{8,1},1,3},{{9,5},0,2}
    };
    common_tree_draft_kv_gather_metadata g{row,3,seg,3,2,3};
    uint32_t out_row[3]={};
    alignas(16) common_tree_draft_kv_gpu_segment out_seg[3]={};
    common_tree_draft_kv_block_table out{out_row,3,out_seg,3,0,0};
    assert(common_tree_draft_kv_block_table_pack(g,&out)==COMMON_TREE_DRAFT_KV_BLOCK_TABLE_OK);
    assert(out.query_count==2 && out.segment_count==3);
    assert(out_row[0]==0 && out_row[1]==2 && out_row[2]==3);
    assert(out_seg[1].page_id==8 && out_seg[1].generation==1 && out_seg[1].lo==1 && out_seg[1].hi==3);
    return 0;
}
