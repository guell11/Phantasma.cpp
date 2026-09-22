#pragma once

#include <cstdint>

enum common_tree_draft_engine_state : uint32_t {
    COMMON_TREE_DRAFT_ENGINE_EMPTY = 0,
    COMMON_TREE_DRAFT_ENGINE_SEEDED,
    COMMON_TREE_DRAFT_ENGINE_EXPANDING,
    COMMON_TREE_DRAFT_ENGINE_SEALED,
    COMMON_TREE_DRAFT_ENGINE_CONSUMED,
    COMMON_TREE_DRAFT_ENGINE_ERROR,
};

enum common_tree_draft_engine_event : uint32_t {
    COMMON_TREE_DRAFT_ENGINE_EVENT_ROOT_SEEDED = 0,
    COMMON_TREE_DRAFT_ENGINE_EVENT_EXPANSION_STEP,
    COMMON_TREE_DRAFT_ENGINE_EVENT_SEAL,
    COMMON_TREE_DRAFT_ENGINE_EVENT_CONSUME,
    COMMON_TREE_DRAFT_ENGINE_EVENT_FAIL,
};

enum common_tree_draft_engine_transition_status : uint32_t {
    COMMON_TREE_DRAFT_ENGINE_TRANSITION_OK = 0,
    COMMON_TREE_DRAFT_ENGINE_TRANSITION_INVALID,
    COMMON_TREE_DRAFT_ENGINE_TRANSITION_TERMINAL,
};

common_tree_draft_engine_transition_status common_tree_draft_engine_transition(
        common_tree_draft_engine_state state,
        common_tree_draft_engine_event event,
        common_tree_draft_engine_state * next_state);

bool common_tree_draft_engine_topology_mutable(common_tree_draft_engine_state state);
bool common_tree_draft_engine_terminal(common_tree_draft_engine_state state);

