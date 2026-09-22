#include "tree-draft-kv-gather.h"

#include <cassert>
#include <vector>

int main() {
    const common_tree_draft_kv_layer_geometry layer[] = {{8,8,COMMON_TREE_DRAFT_KV_LAYOUT_STANDARD}};
    common_tree_draft_kv_page_geometry geom{};
    assert(common_tree_draft_kv_page_geometry_build(4,layer,1,16,&geom)==COMMON_TREE_DRAFT_KV_PAGE_OK);

    const common_tree_draft_kv_branch_span root_spans[] = {{{0,1},0,4},{{1,1},0,2}};
    const common_tree_draft_kv_branch_span child_spans[] = {{{0,1},0,4},{{2,1},0,4}};
    const common_tree_draft_kv_branch_descriptor branches[] = {
        {{10,1},{COMMON_TREE_DRAFT_KV_ID_INVALID,0},-1,5,0,root_spans,2},
        {{11,1},{10,1},3,7,0,child_spans,2},
    };
    int64_t cutoffs[4]={};
    common_tree_draft_kv_visibility_table vis{cutoffs,4,0};
    assert(common_tree_draft_kv_visibility_build(branches,2,vis)==COMMON_TREE_DRAFT_KV_VISIBILITY_OK);

    const common_tree_draft_kv_query queries[]={{0,5},{1,6},{1,3}};
    uint32_t row[4]={};
    common_tree_draft_kv_gather_segment seg[5]={};
    common_tree_draft_kv_gather_metadata out{row,4,seg,5,0,0};
    assert(common_tree_draft_kv_gather_build(queries,3,branches,2,vis,geom,&out)==COMMON_TREE_DRAFT_KV_GATHER_OK);
    assert(out.query_count==3 && out.segment_count==5);
    assert(row[0]==0 && row[1]==2 && row[2]==4 && row[3]==5);
    assert(seg[0].page.id==0 && seg[0].lo==0 && seg[0].hi==4);
    assert(seg[1].page.id==1 && seg[1].hi==2);
    assert(seg[2].page.id==0 && seg[3].page.id==2 && seg[3].hi==3);
    assert(seg[4].page.id==0 && seg[4].hi==4);
    return 0;
}
