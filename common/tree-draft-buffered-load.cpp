#include "tree-draft-buffered-load.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

namespace fs = std::filesystem;

static bool valid_alignment(uint64_t alignment) {
    return alignment != 0 && (alignment & (alignment - 1)) == 0 && alignment <= SIZE_MAX;
}

static std::shared_ptr<void> allocate_aligned(uint64_t length, uint64_t alignment, uint8_t ** data) {
    if (data == nullptr || length == 0 || length > SIZE_MAX || !valid_alignment(alignment)) return {};
    void * ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(static_cast<size_t>(length), static_cast<size_t>(alignment));
    if (ptr == nullptr) return {};
    std::shared_ptr<void> owner(ptr, [](void * p) { _aligned_free(p); });
#else
    const size_t a = static_cast<size_t>(alignment);
    const size_t n = static_cast<size_t>((length + alignment - 1) / alignment * alignment);
    ptr = std::aligned_alloc(a, n);
    if (ptr == nullptr) return {};
    std::shared_ptr<void> owner(ptr, [](void * p) { std::free(p); });
#endif
    *data = static_cast<uint8_t *>(ptr);
    return owner;
}

common_tree_draft_buffered_load_status common_tree_draft_buffered_load(
        const common_tree_draft_model_source & source,
        const common_tree_draft_buffer_interval * intervals,
        size_t interval_count,
        common_tree_draft_buffer_fallback_trigger trigger,
        std::vector<common_tree_draft_buffered_payload> * payloads) {
    if (payloads == nullptr || (interval_count > 0 && intervals == nullptr)) return COMMON_TREE_DRAFT_BUFFERED_LOAD_RANGE;
    payloads->clear();
    if (trigger == COMMON_TREE_DRAFT_BUFFER_TRIGGER_CORRUPT) return COMMON_TREE_DRAFT_BUFFERED_LOAD_TRIGGER;

    std::vector<common_tree_draft_buffer_interval> ordered(intervals, intervals + interval_count);
    for (const auto & interval : ordered) {
        if (interval.path_index >= source.paths.size() || interval.length == 0 || !valid_alignment(interval.alignment) ||
            interval.offset > std::numeric_limits<uint64_t>::max() - interval.length) {
            return !valid_alignment(interval.alignment) ? COMMON_TREE_DRAFT_BUFFERED_LOAD_ALIGNMENT : COMMON_TREE_DRAFT_BUFFERED_LOAD_RANGE;
        }
    }
    std::stable_sort(ordered.begin(), ordered.end(), [](const auto & a, const auto & b) {
        if (a.path_index != b.path_index) return a.path_index < b.path_index;
        return a.offset < b.offset;
    });

    std::vector<common_tree_draft_buffered_payload> temp;
    temp.reserve(interval_count);
    for (const auto & interval : ordered) {
        std::error_code ec;
        const uint64_t file_size = static_cast<uint64_t>(fs::file_size(source.paths[interval.path_index], ec));
        if (ec || interval.offset + interval.length > file_size) return COMMON_TREE_DRAFT_BUFFERED_LOAD_RANGE;

        std::ifstream file(source.paths[interval.path_index], std::ios::binary);
        if (!file) return COMMON_TREE_DRAFT_BUFFERED_LOAD_OPEN;
        file.seekg(static_cast<std::streamoff>(interval.offset), std::ios::beg);
        if (!file) return COMMON_TREE_DRAFT_BUFFERED_LOAD_READ;

        uint8_t * data = nullptr;
        auto owner = allocate_aligned(interval.length, interval.alignment, &data);
        if (!owner) return COMMON_TREE_DRAFT_BUFFERED_LOAD_ALLOC;
        if (interval.length > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) return COMMON_TREE_DRAFT_BUFFERED_LOAD_RANGE;
        file.read(reinterpret_cast<char *>(data), static_cast<std::streamsize>(interval.length));
        if (static_cast<uint64_t>(file.gcount()) != interval.length) return COMMON_TREE_DRAFT_BUFFERED_LOAD_READ;

        common_tree_draft_storage_owner storage = { owner, data, interval.length, interval.alignment, true };
        common_tree_draft_storage_view view = {};
        if (common_tree_draft_storage_view_make(storage, 0, interval.length, true, &view) != COMMON_TREE_DRAFT_STORAGE_VIEW_OK) {
            return COMMON_TREE_DRAFT_BUFFERED_LOAD_ALLOC;
        }
        temp.push_back({view, interval.path_index, interval.offset});
    }
    *payloads = std::move(temp);
    return COMMON_TREE_DRAFT_BUFFERED_LOAD_OK;
}

