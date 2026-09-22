#include "tree-draft-dequant-fallback.h"

#include <cassert>

int main() {
    common_tree_draft_quant_type q{};
    q.id=11; q.block_elems=32; q.block_bytes=18;
    common_tree_draft_fallback_budget budget{1024,128,0};
    common_tree_draft_dequant_fallback_request req{&q,256,COMMON_TREE_DRAFT_DTYPE_F16,true};
    common_tree_draft_dequant_fallback_plan plan{};
    assert(common_tree_draft_dequant_fallback_plan_build(req,budget,&plan)==COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK);
    assert(plan.materialize && plan.output_bytes==512);

    common_tree_draft_dequant_fallback_reservation reservation{};
    assert(common_tree_draft_dequant_fallback_reserve(&budget,plan,&reservation)==COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK);
    assert(budget.committed_bytes==128 && budget.reserved_bytes==512);
    assert(common_tree_draft_dequant_fallback_commit(&budget,&reservation)==COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK);
    assert(budget.committed_bytes==640 && budget.reserved_bytes==0);

    req.logical_elements=200;
    assert(common_tree_draft_dequant_fallback_plan_build(req,budget,&plan)==COMMON_TREE_DRAFT_DEQUANT_FALLBACK_RESOURCE);
    assert(budget.committed_bytes==640 && budget.reserved_bytes==0);

    budget={1024,0,0};
    req.logical_elements=128;
    assert(common_tree_draft_dequant_fallback_plan_build(req,budget,&plan)==COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK);
    assert(common_tree_draft_dequant_fallback_reserve(&budget,plan,&reservation)==COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK);
    assert(common_tree_draft_dequant_fallback_abort(&budget,&reservation)==COMMON_TREE_DRAFT_DEQUANT_FALLBACK_OK);
    assert(budget.committed_bytes==0 && budget.reserved_bytes==0);

    req.policy_permits=false;
    assert(common_tree_draft_dequant_fallback_plan_build(req,budget,&plan)==COMMON_TREE_DRAFT_DEQUANT_FALLBACK_POLICY);
    return 0;
}
