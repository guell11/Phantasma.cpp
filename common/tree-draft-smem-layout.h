#pragma once

#include "tree-draft-smem-carveout.h"

#include <array>
#include <cstddef>
#include <cstdint>

enum common_tree_draft_smem_component_kind : uint32_t {
    COMMON_TREE_DRAFT_SMEM_LOGITS_TILE = 0,
    COMMON_TREE_DRAFT_SMEM_TOKEN_IDS,
    COMMON_TREE_DRAFT_SMEM_PREFIX_META,
    COMMON_TREE_DRAFT_SMEM_REDUCTION_SCRATCH,
    COMMON_TREE_DRAFT_SMEM_USER,
};

struct common_tree_draft_smem_component {
    common_tree_draft_smem_component_kind kind = COMMON_TREE_DRAFT_SMEM_USER;
    uint32_t size_bytes = 0;
    uint32_t alignment = 1;
};

struct common_tree_draft_smem_component_layout {
    common_tree_draft_smem_component_kind kind = COMMON_TREE_DRAFT_SMEM_USER;
    uint32_t offset = 0;
    uint32_t size_bytes = 0;
};

struct common_tree_draft_smem_layout_request {
    uint32_t static_shared_bytes = 0;
    uint32_t selected_shared_limit = 0;
    const common_tree_draft_smem_component * components = nullptr;
    size_t component_count = 0;
};

struct common_tree_draft_smem_layout {
    std::array<common_tree_draft_smem_component_layout, 16> components = {};
    uint8_t component_count = 0;
    uint32_t dynamic_shared_bytes = 0;
    uint32_t total_shared_bytes = 0;
};

enum common_tree_draft_smem_layout_status : uint32_t {
    COMMON_TREE_DRAFT_SMEM_LAYOUT_OK = 0,
    COMMON_TREE_DRAFT_SMEM_LAYOUT_NULL_INPUT,
    COMMON_TREE_DRAFT_SMEM_LAYOUT_TOO_MANY_COMPONENTS,
    COMMON_TREE_DRAFT_SMEM_LAYOUT_BAD_ALIGNMENT,
    COMMON_TREE_DRAFT_SMEM_LAYOUT_OVERFLOW,
    COMMON_TREE_DRAFT_SMEM_LAYOUT_EXCEEDS_LIMIT,
};

common_tree_draft_smem_layout_status common_tree_draft_smem_layout_build(
        const common_tree_draft_smem_layout_request & request,
        common_tree_draft_smem_layout * layout);
