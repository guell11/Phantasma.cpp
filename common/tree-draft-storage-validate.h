#pragma once

#include "tree-draft-alignment.h"
#include "tree-draft-quant-registry.h"
#include "tree-draft-storage-view.h"
#include "tree-draft-tensor-layout.h"

#include <cstdint>
#include <vector>

enum common_tree_draft_endian : uint32_t {
    COMMON_TREE_DRAFT_ENDIAN_LITTLE = 0,
    COMMON_TREE_DRAFT_ENDIAN_BIG,
};

enum common_tree_draft_storage_validation_result : uint32_t {
    COMMON_TREE_DRAFT_STORAGE_VALID = 0,
    COMMON_TREE_DRAFT_STORAGE_RECOVERABLE_COPY,
    COMMON_TREE_DRAFT_STORAGE_CORRUPT,
    COMMON_TREE_DRAFT_STORAGE_UNSUPPORTED,
};

struct common_tree_draft_storage_validation {
    common_tree_draft_storage_validation_result result = COMMON_TREE_DRAFT_STORAGE_CORRUPT;
    uint64_t expected_span = 0;
    uint64_t required_alignment = 1;
};

common_tree_draft_storage_validation common_tree_draft_storage_validate(
        common_tree_draft_endian source_endian,
        const common_tree_draft_tensor_layout & layout,
        const common_tree_draft_quant_type & quant,
        const common_tree_draft_storage_view & view,
        const common_tree_draft_alignment_table & alignments,
        common_tree_draft_buffer_class buffer_class,
        bool copy_fallback_allowed);

