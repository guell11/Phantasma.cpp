#include "tree-draft-storage-validate.h"

#include <cassert>
#include <memory>
#include <vector>

int main() {
    common_tree_draft_alignment_table table = {};
    assert(common_tree_draft_alignment_build({16,64,32,64}, &table) == COMMON_TREE_DRAFT_ALIGNMENT_OK);

    auto backing = std::make_shared<std::vector<uint8_t>>(256);
    std::shared_ptr<void> life(backing, backing->data());
    common_tree_draft_storage_owner owner = { life, backing->data(), 256, 64, true };
    common_tree_draft_storage_view view = {};
    assert(common_tree_draft_storage_view_make(owner, 0, 64, true, &view) == COMMON_TREE_DRAFT_STORAGE_VIEW_OK);

    common_tree_draft_tensor_layout layout = {
        COMMON_TREE_DRAFT_LAYOUT_DENSE,
        {COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {COMMON_TREE_DRAFT_AXIS_INPUT, COMMON_TREE_DRAFT_AXIS_OUTPUT},
        {0,1}, {4,8}, {4,8}, {}, {8,1}, false,
    };
    const auto * f16 = common_tree_draft_quant_find_ggml(GGML_TYPE_F16);
    assert(f16 != nullptr);
    auto ok = common_tree_draft_storage_validate(COMMON_TREE_DRAFT_ENDIAN_LITTLE, layout, *f16, view, table, COMMON_TREE_DRAFT_BUFFER_METADATA, true);
    assert(ok.result == COMMON_TREE_DRAFT_STORAGE_VALID && ok.expected_span == 64);

    common_tree_draft_storage_view misaligned = {};
    assert(common_tree_draft_storage_view_make(owner, 2, 64, true, &misaligned) == COMMON_TREE_DRAFT_STORAGE_VIEW_OK);
    auto recover = common_tree_draft_storage_validate(COMMON_TREE_DRAFT_ENDIAN_LITTLE, layout, *f16, misaligned, table, COMMON_TREE_DRAFT_BUFFER_METADATA, true);
    assert(recover.result == COMMON_TREE_DRAFT_STORAGE_RECOVERABLE_COPY);
    auto no_copy = common_tree_draft_storage_validate(COMMON_TREE_DRAFT_ENDIAN_LITTLE, layout, *f16, misaligned, table, COMMON_TREE_DRAFT_BUFFER_METADATA, false);
    assert(no_copy.result == COMMON_TREE_DRAFT_STORAGE_CORRUPT);

    common_tree_draft_storage_view bad_span = view;
    bad_span.length = 62;
    assert(common_tree_draft_storage_validate(COMMON_TREE_DRAFT_ENDIAN_LITTLE, layout, *f16, bad_span, table, COMMON_TREE_DRAFT_BUFFER_METADATA, true).result == COMMON_TREE_DRAFT_STORAGE_CORRUPT);
    assert(common_tree_draft_storage_validate(COMMON_TREE_DRAFT_ENDIAN_BIG, layout, *f16, view, table, COMMON_TREE_DRAFT_BUFFER_METADATA, true).result == COMMON_TREE_DRAFT_STORAGE_UNSUPPORTED);
    return 0;
}

