#include "tree-draft-gptq-reindex.h"

#include <cassert>
#include <vector>

static common_tree_draft_packed_linear_view base_view() {
    common_tree_draft_packed_linear_view view;
    view.format = COMMON_TREE_DRAFT_PACKED_LINEAR_GPTQ;
    view.in_features = 8;
    view.out_features = 4;
    view.group_size = 4;
    view.group_count = 2;
    return view;
}

int main() {
    common_tree_draft_gptq_descriptor quant;
    quant.bits = 4;
    quant.group_size = 4;

    // Ordinary GPTQ has the canonical contiguous group mapping and identity
    // stable grouping order.
    auto view = base_view();
    common_tree_draft_gptq_reindex_descriptor direct;
    assert(common_tree_draft_gptq_build_reindex(quant, view, nullptr, &direct) == COMMON_TREE_DRAFT_GPTQ_REINDEX_OK);
    assert(!direct.activation_order);
    assert((direct.group_for_logical_column == std::vector<uint32_t>{0,0,0,0,1,1,1,1}));
    assert((direct.grouped_to_logical == std::vector<uint32_t>{0,1,2,3,4,5,6,7}));
    assert(direct.logical_to_grouped == direct.grouped_to_logical);

    // desc_act preserves logical column identity while exposing a stable
    // metadata-only order grouped by the exact g_idx group lookup.
    quant.desc_act = true;
    view.g_idx = "layer.g_idx";
    const std::vector<int64_t> g_idx = {1,0,1,0,1,0,1,0};
    common_tree_draft_gptq_reindex_descriptor act;
    assert(common_tree_draft_gptq_build_reindex(quant, view, &g_idx, &act) == COMMON_TREE_DRAFT_GPTQ_REINDEX_OK);
    assert(act.activation_order);
    assert((act.group_for_logical_column == std::vector<uint32_t>{1,0,1,0,1,0,1,0}));
    assert((act.grouped_to_logical == std::vector<uint32_t>{1,3,5,7,0,2,4,6}));
    for (uint32_t logical = 0; logical < 8; ++logical) {
        assert(act.grouped_to_logical[act.logical_to_grouped[logical]] == logical);
    }

    // Failures are explicit and do not partially replace an existing output.
    common_tree_draft_gptq_reindex_descriptor sentinel = act;
    auto missing = view;
    missing.g_idx.reset();
    assert(common_tree_draft_gptq_build_reindex(quant, missing, &g_idx, &sentinel) == COMMON_TREE_DRAFT_GPTQ_REINDEX_MISSING_GIDX);
    assert(sentinel.group_for_logical_column == act.group_for_logical_column);

    const std::vector<int64_t> short_idx = {0,1};
    assert(common_tree_draft_gptq_build_reindex(quant, view, &short_idx, &sentinel) == COMMON_TREE_DRAFT_GPTQ_REINDEX_LENGTH);
    const std::vector<int64_t> bad_group = {0,0,0,0,1,1,1,2};
    assert(common_tree_draft_gptq_build_reindex(quant, view, &bad_group, &sentinel) == COMMON_TREE_DRAFT_GPTQ_REINDEX_GROUP_RANGE);
    const std::vector<int64_t> negative_group = {0,0,0,0,1,1,1,-1};
    assert(common_tree_draft_gptq_build_reindex(quant, view, &negative_group, &sentinel) == COMMON_TREE_DRAFT_GPTQ_REINDEX_GROUP_RANGE);

    auto wrong_format = view;
    wrong_format.format = COMMON_TREE_DRAFT_PACKED_LINEAR_AWQ;
    assert(common_tree_draft_gptq_build_reindex(quant, wrong_format, &g_idx, &sentinel) == COMMON_TREE_DRAFT_GPTQ_REINDEX_INVALID);
    return 0;
}
