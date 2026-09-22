#pragma once

#include "tree-draft-forest.h"
#include "tree-draft-topology.h"

#include <cstdint>

enum common_tree_draft_validator_error : int32_t {
    COMMON_TREE_DRAFT_VALIDATOR_OK = 0,
    COMMON_TREE_DRAFT_VALIDATOR_INVALID_TOPOLOGY,
    COMMON_TREE_DRAFT_VALIDATOR_NEGATIVE_ENTRY_COUNT,
    COMMON_TREE_DRAFT_VALIDATOR_NULL_OFFSETS,
    COMMON_TREE_DRAFT_VALIDATOR_OFFSET_START,
    COMMON_TREE_DRAFT_VALIDATOR_NEGATIVE_OFFSET,
    COMMON_TREE_DRAFT_VALIDATOR_OFFSET_ORDER,
    COMMON_TREE_DRAFT_VALIDATOR_OFFSET_RANGE,
    COMMON_TREE_DRAFT_VALIDATOR_OFFSET_END,
};

struct common_tree_draft_validation_result {
    common_tree_draft_validator_error error;
    common_tree_draft_topology_error topology_error;
};

// Validates the accepted ID_001 topology contract together with the ID_004
// packed-forest offset descriptor. All malformed-input checks are always on.
common_tree_draft_validation_result common_tree_draft_validate(
        const common_tree_draft_topology & topology,
        const common_tree_draft_forest_offsets & forest);

const char * common_tree_draft_validator_error_name(common_tree_draft_validator_error error);
