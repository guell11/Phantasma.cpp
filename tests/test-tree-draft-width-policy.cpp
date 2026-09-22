#include "tree-draft-width-policy.h"

#include <cassert>
#include <cmath>

int main() {
    const common_tree_draft_candidate peaked[] = {
        { 1, 3.0f, 0.98f }, { 2, 2.0f, 0.01f }, { 3, 1.0f, 0.01f },
    };
    const common_tree_draft_candidate flat[] = {
        { 1, 3.0f, 1.0f/3.0f }, { 2, 2.0f, 1.0f/3.0f }, { 3, 1.0f, 1.0f/3.0f },
    };
    uint32_t width = 0;
    double h = 0.0;
    assert(common_tree_draft_entropy_width(peaked, 3, 1, 5, &width, &h) == COMMON_TREE_DRAFT_WIDTH_OK);
    assert(width >= 1 && width < 5 && h < 0.2);
    assert(common_tree_draft_entropy_width(flat, 3, 1, 5, &width, &h) == COMMON_TREE_DRAFT_WIDTH_OK);
    assert(width == 5 && std::fabs(h - 1.0) < 1e-6);

    const common_tree_draft_candidate singleton[] = { { 7, 1.0f, 0.2f } };
    assert(common_tree_draft_entropy_width(singleton, 1, 2, 6, &width, &h) == COMMON_TREE_DRAFT_WIDTH_OK);
    assert(width == 2 && h == 0.0);

    common_tree_draft_width_override override_result = {};
    assert(common_tree_draft_confidence_width(peaked, 3, 0.1, 0.8, 5, &override_result) == COMMON_TREE_DRAFT_WIDTH_OK);
    assert(override_result.kind == COMMON_TREE_DRAFT_WIDTH_OVERRIDE && override_result.width == 1);
    assert(common_tree_draft_confidence_width(flat, 3, 0.1, 0.8, 5, &override_result) == COMMON_TREE_DRAFT_WIDTH_OK);
    assert(override_result.kind == COMMON_TREE_DRAFT_WIDTH_OVERRIDE && override_result.width == 5);

    const common_tree_draft_candidate mid[] = { { 1, 2.0f, 0.65f }, { 2, 1.0f, 0.35f } };
    assert(common_tree_draft_confidence_width(mid, 2, 0.1, 0.8, 5, &override_result) == COMMON_TREE_DRAFT_WIDTH_OK);
    assert(override_result.kind == COMMON_TREE_DRAFT_WIDTH_NO_OVERRIDE);
    return 0;
}

