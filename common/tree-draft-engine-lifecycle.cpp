#include "tree-draft-engine-lifecycle.h"

bool common_tree_draft_engine_terminal(common_tree_draft_engine_state state) {
    return state == COMMON_TREE_DRAFT_ENGINE_CONSUMED || state == COMMON_TREE_DRAFT_ENGINE_ERROR;
}

bool common_tree_draft_engine_topology_mutable(common_tree_draft_engine_state state) {
    return state == COMMON_TREE_DRAFT_ENGINE_SEEDED || state == COMMON_TREE_DRAFT_ENGINE_EXPANDING;
}

common_tree_draft_engine_transition_status common_tree_draft_engine_transition(
        common_tree_draft_engine_state state,
        common_tree_draft_engine_event event,
        common_tree_draft_engine_state * next_state) {
    if (next_state == nullptr) {
        return COMMON_TREE_DRAFT_ENGINE_TRANSITION_INVALID;
    }
    if (common_tree_draft_engine_terminal(state)) {
        return COMMON_TREE_DRAFT_ENGINE_TRANSITION_TERMINAL;
    }

    common_tree_draft_engine_state next = state;
    bool valid = true;

    if (event == COMMON_TREE_DRAFT_ENGINE_EVENT_FAIL) {
        next = COMMON_TREE_DRAFT_ENGINE_ERROR;
    } else {
        switch (state) {
            case COMMON_TREE_DRAFT_ENGINE_EMPTY:
                if (event == COMMON_TREE_DRAFT_ENGINE_EVENT_ROOT_SEEDED) {
                    next = COMMON_TREE_DRAFT_ENGINE_SEEDED;
                } else {
                    valid = false;
                }
                break;
            case COMMON_TREE_DRAFT_ENGINE_SEEDED:
                if (event == COMMON_TREE_DRAFT_ENGINE_EVENT_EXPANSION_STEP) {
                    next = COMMON_TREE_DRAFT_ENGINE_EXPANDING;
                } else if (event == COMMON_TREE_DRAFT_ENGINE_EVENT_SEAL) {
                    next = COMMON_TREE_DRAFT_ENGINE_SEALED;
                } else {
                    valid = false;
                }
                break;
            case COMMON_TREE_DRAFT_ENGINE_EXPANDING:
                if (event == COMMON_TREE_DRAFT_ENGINE_EVENT_EXPANSION_STEP) {
                    next = COMMON_TREE_DRAFT_ENGINE_EXPANDING;
                } else if (event == COMMON_TREE_DRAFT_ENGINE_EVENT_SEAL) {
                    next = COMMON_TREE_DRAFT_ENGINE_SEALED;
                } else {
                    valid = false;
                }
                break;
            case COMMON_TREE_DRAFT_ENGINE_SEALED:
                if (event == COMMON_TREE_DRAFT_ENGINE_EVENT_CONSUME) {
                    next = COMMON_TREE_DRAFT_ENGINE_CONSUMED;
                } else {
                    valid = false;
                }
                break;
            case COMMON_TREE_DRAFT_ENGINE_CONSUMED:
            case COMMON_TREE_DRAFT_ENGINE_ERROR:
                return COMMON_TREE_DRAFT_ENGINE_TRANSITION_TERMINAL;
        }
    }

    if (!valid) {
        return COMMON_TREE_DRAFT_ENGINE_TRANSITION_INVALID;
    }
    *next_state = next;
    return COMMON_TREE_DRAFT_ENGINE_TRANSITION_OK;
}

