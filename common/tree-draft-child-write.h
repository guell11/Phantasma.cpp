#pragma once

#include "tree-draft-append.h"
#include "tree-draft-multi-sample.h"
#include "tree-draft-path.h"

#include <cstddef>
#include <cstdint>

enum common_tree_draft_child_write_status : uint32_t {
    COMMON_TREE_DRAFT_CHILD_WRITE_OK = 0,
    COMMON_TREE_DRAFT_CHILD_WRITE_NULL_BUFFER,
    COMMON_TREE_DRAFT_CHILD_WRITE_PHASE,
    COMMON_TREE_DRAFT_CHILD_WRITE_RANGE,
    COMMON_TREE_DRAFT_CHILD_WRITE_DEPTH,
    COMMON_TREE_DRAFT_CHILD_WRITE_PATH,
    COMMON_TREE_DRAFT_CHILD_WRITE_PROBABILITY,
    COMMON_TREE_DRAFT_CHILD_WRITE_NUMERIC,
    COMMON_TREE_DRAFT_CHILD_WRITE_APPEND,
};

common_tree_draft_child_write_status common_tree_draft_fill_child_metadata(
        common_tree_draft_arena * arena,
        common_tree_draft_append_transaction * transaction,
        uint32_t parent_index,
        uint32_t relative_child_index,
        const common_tree_draft_child_sample & sample,
        common_tree_draft_node * child);

common_tree_draft_child_write_status common_tree_draft_write_child_probability(
        common_tree_draft_arena * arena,
        common_tree_draft_append_transaction * transaction,
        uint32_t relative_child_index,
        const common_tree_draft_child_sample & sample,
        const double * cumulative_before_append,
        size_t cumulative_capacity,
        float * local_probability,
        size_t local_capacity,
        double * cumulative_logp,
        size_t cumulative_out_capacity);

