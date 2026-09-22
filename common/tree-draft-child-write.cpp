#include "tree-draft-child-write.h"

#include <cmath>
#include <limits>

common_tree_draft_child_write_status common_tree_draft_fill_child_metadata(
        common_tree_draft_arena * arena,
        common_tree_draft_append_transaction * transaction,
        uint32_t parent_index,
        uint32_t relative_child_index,
        const common_tree_draft_child_sample & sample,
        common_tree_draft_node * child) {
    if (arena == nullptr || transaction == nullptr || child == nullptr) return COMMON_TREE_DRAFT_CHILD_WRITE_NULL_BUFFER;
    if (transaction->phase != COMMON_TREE_DRAFT_APPEND_RESERVED) return COMMON_TREE_DRAFT_CHILD_WRITE_PHASE;
    if (parent_index >= transaction->base_node_count || relative_child_index >= transaction->total_children) return COMMON_TREE_DRAFT_CHILD_WRITE_RANGE;
    const common_tree_draft_node & parent = arena->nodes[parent_index];
    if (parent.depth < 0 || parent.depth == std::numeric_limits<int32_t>::max()) return COMMON_TREE_DRAFT_CHILD_WRITE_DEPTH;
    const uint32_t child_depth = static_cast<uint32_t>(parent.depth + 1);
    uint64_t path_id = 0;
    if (common_tree_draft_path_child(parent.path_id, sample.token, sample.draw_ordinal, child_depth, &path_id) != COMMON_TREE_DRAFT_PATH_OK) {
        return COMMON_TREE_DRAFT_CHILD_WRITE_PATH;
    }
    *child = {
        static_cast<int32_t>(parent_index),
        sample.token,
        static_cast<int32_t>(child_depth),
        sample.log_conditional_probability,
        path_id,
        COMMON_TREE_DRAFT_NODE_FLAG_NONE,
    };
    return COMMON_TREE_DRAFT_CHILD_WRITE_OK;
}

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
        size_t cumulative_out_capacity) {
    if (arena == nullptr || transaction == nullptr || cumulative_before_append == nullptr || local_probability == nullptr || cumulative_logp == nullptr) {
        return COMMON_TREE_DRAFT_CHILD_WRITE_NULL_BUFFER;
    }
    if (transaction->phase != COMMON_TREE_DRAFT_APPEND_RESERVED || relative_child_index >= transaction->total_children) return COMMON_TREE_DRAFT_CHILD_WRITE_PHASE;
    const uint32_t absolute = transaction->base_node_count + relative_child_index;
    if (absolute >= local_capacity || absolute >= cumulative_out_capacity) return COMMON_TREE_DRAFT_CHILD_WRITE_RANGE;
    const common_tree_draft_node & child = arena->nodes[absolute];
    if (child.parent < 0 || static_cast<size_t>(child.parent) >= cumulative_capacity) return COMMON_TREE_DRAFT_CHILD_WRITE_RANGE;
    const double p = sample.conditional_probability;
    if (!(p > 0.0) || p > 1.0 || !std::isfinite(p)) return COMMON_TREE_DRAFT_CHILD_WRITE_PROBABILITY;
    const double logp = std::log(p);
    if (!std::isfinite(logp) || std::fabs(logp - static_cast<double>(sample.log_conditional_probability)) > 1e-5) {
        return COMMON_TREE_DRAFT_CHILD_WRITE_PROBABILITY;
    }
    const double cumulative = cumulative_before_append[child.parent] + logp;
    if (!std::isfinite(cumulative)) return COMMON_TREE_DRAFT_CHILD_WRITE_NUMERIC;
    arena->nodes[absolute].logp = static_cast<float>(logp);
    local_probability[absolute] = static_cast<float>(p);
    cumulative_logp[absolute] = cumulative;
    return COMMON_TREE_DRAFT_CHILD_WRITE_OK;
}

