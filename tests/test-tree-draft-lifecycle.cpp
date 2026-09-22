#include "tree-draft-lifecycle.h"

#include <cassert>
#include <cstring>

static void test_legal_transitions() {
    assert(common_tree_draft_lifecycle_transition_is_valid(
            COMMON_TREE_DRAFT_LIFECYCLE_CREATED,
            COMMON_TREE_DRAFT_LIFECYCLE_QUEUED));
    assert(common_tree_draft_lifecycle_transition_is_valid(
            COMMON_TREE_DRAFT_LIFECYCLE_QUEUED,
            COMMON_TREE_DRAFT_LIFECYCLE_RUNNING));
    assert(common_tree_draft_lifecycle_transition_is_valid(
            COMMON_TREE_DRAFT_LIFECYCLE_RUNNING,
            COMMON_TREE_DRAFT_LIFECYCLE_STREAMING));

    const common_tree_draft_lifecycle_state active[] = {
        COMMON_TREE_DRAFT_LIFECYCLE_QUEUED,
        COMMON_TREE_DRAFT_LIFECYCLE_RUNNING,
        COMMON_TREE_DRAFT_LIFECYCLE_STREAMING,
    };
    const common_tree_draft_lifecycle_state terminal[] = {
        COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED,
        COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED,
        COMMON_TREE_DRAFT_LIFECYCLE_FAILED,
    };

    for (const auto from : active) {
        for (const auto to : terminal) {
            assert(common_tree_draft_lifecycle_transition_is_valid(from, to));
        }
    }
}

static void test_illegal_transitions_and_terminal_states() {
    const common_tree_draft_lifecycle_state states[] = {
        COMMON_TREE_DRAFT_LIFECYCLE_CREATED,
        COMMON_TREE_DRAFT_LIFECYCLE_QUEUED,
        COMMON_TREE_DRAFT_LIFECYCLE_RUNNING,
        COMMON_TREE_DRAFT_LIFECYCLE_STREAMING,
        COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED,
        COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED,
        COMMON_TREE_DRAFT_LIFECYCLE_FAILED,
    };
    const common_tree_draft_lifecycle_state terminal_states[] = {
        COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED,
        COMMON_TREE_DRAFT_LIFECYCLE_CANCELLED,
        COMMON_TREE_DRAFT_LIFECYCLE_FAILED,
    };

    for (const auto terminal : terminal_states) {
        for (const auto to : states) {
            assert(!common_tree_draft_lifecycle_transition_is_valid(terminal, to));
        }
    }

    assert(!common_tree_draft_lifecycle_transition_is_valid(
            COMMON_TREE_DRAFT_LIFECYCLE_CREATED,
            COMMON_TREE_DRAFT_LIFECYCLE_COMPLETED));
    assert(!common_tree_draft_lifecycle_transition_is_valid(
            COMMON_TREE_DRAFT_LIFECYCLE_QUEUED,
            COMMON_TREE_DRAFT_LIFECYCLE_STREAMING));
    assert(!common_tree_draft_lifecycle_transition_is_valid(
            COMMON_TREE_DRAFT_LIFECYCLE_RUNNING,
            COMMON_TREE_DRAFT_LIFECYCLE_CREATED));
}

static void test_state_names() {
    assert(std::strcmp(common_tree_draft_lifecycle_state_name(COMMON_TREE_DRAFT_LIFECYCLE_CREATED), "created") == 0);
    assert(std::strcmp(common_tree_draft_lifecycle_state_name(COMMON_TREE_DRAFT_LIFECYCLE_FAILED), "failed") == 0);
    assert(std::strcmp(common_tree_draft_lifecycle_state_name((common_tree_draft_lifecycle_state) -1), "unknown") == 0);
}

int main() {
    test_legal_transitions();
    test_illegal_transitions_and_terminal_states();
    test_state_names();
    return 0;
}
