#include "tree-draft-engine-lifecycle.h"

#include <cassert>

static void expect_transition(
        common_tree_draft_engine_state from,
        common_tree_draft_engine_event event,
        common_tree_draft_engine_state expected) {
    common_tree_draft_engine_state next = COMMON_TREE_DRAFT_ENGINE_ERROR;
    assert(common_tree_draft_engine_transition(from, event, &next) == COMMON_TREE_DRAFT_ENGINE_TRANSITION_OK);
    assert(next == expected);
}

int main() {
    expect_transition(COMMON_TREE_DRAFT_ENGINE_EMPTY, COMMON_TREE_DRAFT_ENGINE_EVENT_ROOT_SEEDED, COMMON_TREE_DRAFT_ENGINE_SEEDED);
    expect_transition(COMMON_TREE_DRAFT_ENGINE_SEEDED, COMMON_TREE_DRAFT_ENGINE_EVENT_EXPANSION_STEP, COMMON_TREE_DRAFT_ENGINE_EXPANDING);
    expect_transition(COMMON_TREE_DRAFT_ENGINE_EXPANDING, COMMON_TREE_DRAFT_ENGINE_EVENT_EXPANSION_STEP, COMMON_TREE_DRAFT_ENGINE_EXPANDING);
    expect_transition(COMMON_TREE_DRAFT_ENGINE_EXPANDING, COMMON_TREE_DRAFT_ENGINE_EVENT_SEAL, COMMON_TREE_DRAFT_ENGINE_SEALED);
    expect_transition(COMMON_TREE_DRAFT_ENGINE_SEALED, COMMON_TREE_DRAFT_ENGINE_EVENT_CONSUME, COMMON_TREE_DRAFT_ENGINE_CONSUMED);
    expect_transition(COMMON_TREE_DRAFT_ENGINE_SEEDED, COMMON_TREE_DRAFT_ENGINE_EVENT_SEAL, COMMON_TREE_DRAFT_ENGINE_SEALED);
    expect_transition(COMMON_TREE_DRAFT_ENGINE_EMPTY, COMMON_TREE_DRAFT_ENGINE_EVENT_FAIL, COMMON_TREE_DRAFT_ENGINE_ERROR);

    common_tree_draft_engine_state next = COMMON_TREE_DRAFT_ENGINE_EMPTY;
    assert(common_tree_draft_engine_transition(COMMON_TREE_DRAFT_ENGINE_EMPTY, COMMON_TREE_DRAFT_ENGINE_EVENT_CONSUME, &next) == COMMON_TREE_DRAFT_ENGINE_TRANSITION_INVALID);
    assert(common_tree_draft_engine_transition(COMMON_TREE_DRAFT_ENGINE_SEALED, COMMON_TREE_DRAFT_ENGINE_EVENT_EXPANSION_STEP, &next) == COMMON_TREE_DRAFT_ENGINE_TRANSITION_INVALID);
    assert(common_tree_draft_engine_transition(COMMON_TREE_DRAFT_ENGINE_CONSUMED, COMMON_TREE_DRAFT_ENGINE_EVENT_FAIL, &next) == COMMON_TREE_DRAFT_ENGINE_TRANSITION_TERMINAL);
    assert(common_tree_draft_engine_transition(COMMON_TREE_DRAFT_ENGINE_ERROR, COMMON_TREE_DRAFT_ENGINE_EVENT_ROOT_SEEDED, &next) == COMMON_TREE_DRAFT_ENGINE_TRANSITION_TERMINAL);
    assert(common_tree_draft_engine_transition(COMMON_TREE_DRAFT_ENGINE_EMPTY, COMMON_TREE_DRAFT_ENGINE_EVENT_ROOT_SEEDED, nullptr) == COMMON_TREE_DRAFT_ENGINE_TRANSITION_INVALID);

    assert(!common_tree_draft_engine_topology_mutable(COMMON_TREE_DRAFT_ENGINE_EMPTY));
    assert(common_tree_draft_engine_topology_mutable(COMMON_TREE_DRAFT_ENGINE_SEEDED));
    assert(common_tree_draft_engine_topology_mutable(COMMON_TREE_DRAFT_ENGINE_EXPANDING));
    assert(!common_tree_draft_engine_topology_mutable(COMMON_TREE_DRAFT_ENGINE_SEALED));
    assert(common_tree_draft_engine_terminal(COMMON_TREE_DRAFT_ENGINE_CONSUMED));
    assert(common_tree_draft_engine_terminal(COMMON_TREE_DRAFT_ENGINE_ERROR));
    return 0;
}

