#include "tree-draft-gguf-mmap-plan.h"

#include <algorithm>
#include <filesystem>
#include <limits>

namespace fs = std::filesystem;

namespace {

common_tree_draft_gguf_mmap_plan_result fail(
        common_tree_draft_gguf_mmap_plan_error error,
        size_t source_path_index,
        const char * message) {
    common_tree_draft_gguf_mmap_plan_result result;
    result.error = error;
    result.source_path_index = source_path_index;
    result.message = message;
    return result;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t & out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool align_up(uint64_t value, uint64_t alignment, uint64_t & out) {
    if (alignment == 0) return false;
    const uint64_t rem = value % alignment;
    if (rem == 0) { out = value; return true; }
    return checked_add(value, alignment - rem, out);
}

struct candidate {
    uint16_t shard_index = 0;
    size_t source_path_index = 0;
    std::string source_path;
    size_t logical_tensor_index = 0;
    std::string tensor_name;
    uint64_t tensor_start = 0;
    uint64_t tensor_size = 0;
    uint64_t page_start = 0;
    uint64_t page_end = 0;
};

}

common_tree_draft_gguf_mmap_plan_result common_tree_draft_gguf_plan_mmap_regions(
        const common_tree_draft_model_source & source,
        const common_tree_draft_gguf_mmap_policy & policy) {
    if (policy.page_size == 0 || policy.max_region_size == 0 || policy.max_region_size < policy.page_size) {
        return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_INVALID_POLICY, 0, "invalid mmap planning policy");
    }

    const auto shards = common_tree_draft_gguf_resolve_shards(source);
    if (!shards) {
        return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_SHARD_ERROR, shards.source_path_index, shards.message.c_str());
    }

    std::vector<uint64_t> file_sizes(source.paths.size(), 0);
    std::vector<uint64_t> data_bases(source.paths.size(), 0);
    for (size_t source_path_index : shards.set.source_path_indices) {
        const auto directory = common_tree_draft_gguf_tensor_directory_parse(source, source_path_index);
        if (!directory) {
            return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_TENSOR_ERROR, source_path_index, directory.message.c_str());
        }
        data_bases[source_path_index] = directory.directory.data_base;

        std::error_code ec;
        const uintmax_t native_size = fs::file_size(source.paths[source_path_index], ec);
        if (ec || native_size > std::numeric_limits<uint64_t>::max()) {
            return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_FILE_ERROR, source_path_index, "failed to obtain shard file size");
        }
        file_sizes[source_path_index] = static_cast<uint64_t>(native_size);
    }

    std::vector<candidate> candidates;
    candidates.reserve(shards.set.tensors.size());
    for (size_t logical_index = 0; logical_index < shards.set.tensors.size(); ++logical_index) {
        const auto & tensor = shards.set.tensors[logical_index];
        candidate c;
        c.shard_index = tensor.shard_index;
        c.source_path_index = tensor.source_path_index;
        c.source_path = tensor.source_path;
        c.logical_tensor_index = logical_index;
        c.tensor_name = tensor.descriptor.name;
        c.tensor_size = tensor.descriptor.storage_size;

        if (!checked_add(data_bases[c.source_path_index], tensor.descriptor.offset, c.tensor_start)) {
            return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OVERFLOW, c.source_path_index, "tensor file offset overflows");
        }
        uint64_t tensor_end = 0;
        if (!checked_add(c.tensor_start, c.tensor_size, tensor_end)) {
            return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OVERFLOW, c.source_path_index, "tensor end offset overflows");
        }
        if (tensor_end > file_sizes[c.source_path_index]) {
            return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OUT_OF_BOUNDS, c.source_path_index, "tensor span exceeds backing shard");
        }
        c.page_start = c.tensor_start - c.tensor_start % policy.page_size;
        if (!align_up(tensor_end, policy.page_size, c.page_end)) {
            return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OVERFLOW, c.source_path_index, "page alignment overflows");
        }
        if (c.page_end > file_sizes[c.source_path_index]) {
            return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OUT_OF_BOUNDS, c.source_path_index, "page-aligned mapping exceeds backing shard");
        }
        if (c.page_end - c.page_start > policy.max_region_size) {
            return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_INVALID_POLICY, c.source_path_index, "single tensor mapping exceeds max_region_size");
        }
        candidates.push_back(std::move(c));
    }

    std::sort(candidates.begin(), candidates.end(), [](const candidate & a, const candidate & b) {
        if (a.shard_index != b.shard_index) return a.shard_index < b.shard_index;
        if (a.page_start != b.page_start) return a.page_start < b.page_start;
        if (a.page_end != b.page_end) return a.page_end < b.page_end;
        return a.logical_tensor_index < b.logical_tensor_index;
    });

    common_tree_draft_gguf_mmap_plan_result result;
    result.plan.policy = policy;
    for (const auto & c : candidates) {
        bool merge = false;
        if (policy.coalesce && !result.plan.regions.empty()) {
            const auto & previous = result.plan.regions.back();
            uint64_t previous_end = 0;
            if (!checked_add(previous.file_offset, previous.length, previous_end)) {
                return fail(COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OVERFLOW, c.source_path_index, "existing region end overflows");
            }
            if (previous.shard_index == c.shard_index && previous.source_path_index == c.source_path_index) {
                const uint64_t gap = c.page_start > previous_end ? c.page_start - previous_end : 0;
                const uint64_t merged_end = std::max(previous_end, c.page_end);
                const uint64_t merged_size = merged_end - previous.file_offset;
                merge = gap <= policy.max_gap && merged_size <= policy.max_region_size;
            }
        }

        if (!merge) {
            common_tree_draft_gguf_mmap_region region;
            region.shard_index = c.shard_index;
            region.source_path_index = c.source_path_index;
            region.source_path = c.source_path;
            region.file_offset = c.page_start;
            region.length = c.page_end - c.page_start;
            result.plan.regions.push_back(std::move(region));
        } else {
            auto & region = result.plan.regions.back();
            const uint64_t new_end = std::max(region.file_offset + region.length, c.page_end);
            region.length = new_end - region.file_offset;
        }

        auto & region = result.plan.regions.back();
        common_tree_draft_gguf_mmap_tensor_binding binding;
        binding.logical_tensor_index = c.logical_tensor_index;
        binding.tensor_name = c.tensor_name;
        binding.file_offset = c.tensor_start;
        binding.byte_size = c.tensor_size;
        binding.intra_map_offset = c.tensor_start - region.file_offset;
        region.tensors.push_back(std::move(binding));
    }
    return result;
}

const char * common_tree_draft_gguf_mmap_plan_error_name(common_tree_draft_gguf_mmap_plan_error error) {
    switch (error) {
        case COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OK: return "ok";
        case COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_INVALID_POLICY: return "invalid_policy";
        case COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_SHARD_ERROR: return "shard_error";
        case COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_TENSOR_ERROR: return "tensor_error";
        case COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_FILE_ERROR: return "file_error";
        case COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OVERFLOW: return "overflow";
        case COMMON_TREE_DRAFT_GGUF_MMAP_PLAN_OUT_OF_BOUNDS: return "out_of_bounds";
    }
    return "unknown";
}
