#include "tree-draft-model-source.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>

namespace fs = std::filesystem;
using json = nlohmann::json;

static constexpr uint64_t SAFETENSORS_MAX_PROBE_HEADER = 64ull * 1024 * 1024;

static bool ends_with(const std::string & value, const char * suffix) {
    const size_t suffix_len = std::char_traits<char>::length(suffix);
    return value.size() >= suffix_len && value.compare(value.size() - suffix_len, suffix_len, suffix) == 0;
}

static uint64_t read_u64_le(const std::array<unsigned char, 8> & bytes) {
    uint64_t value = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

static bool read_exact(std::ifstream & file, void * dst, size_t size) {
    if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    file.read(static_cast<char *>(dst), static_cast<std::streamsize>(size));
    return static_cast<size_t>(file.gcount()) == size;
}

static uint64_t file_size_or_zero(const fs::path & path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    return ec ? 0 : static_cast<uint64_t>(size);
}

static void add_evidence(
        common_tree_draft_model_source & result,
        const fs::path & path,
        common_tree_draft_model_probe_signal signal,
        uint64_t file_size = 0,
        uint64_t bytes_read = 0,
        uint64_t header_size = 0) {
    result.evidence.push_back({
        path.lexically_normal().generic_string(),
        signal,
        file_size,
        bytes_read,
        header_size,
    });
}

static bool collect_safetensors_index(
        const fs::path & index_path,
        std::vector<fs::path> & candidates,
        common_tree_draft_model_source & result) {
    std::ifstream file(index_path, std::ios::binary);
    if (!file) {
        add_evidence(result, index_path, COMMON_TREE_DRAFT_MODEL_PROBE_MISSING);
        return false;
    }

    json index;
    try {
        file >> index;
    } catch (const json::exception &) {
        add_evidence(result, index_path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size_or_zero(index_path));
        return false;
    }

    const auto weight_map_it = index.find("weight_map");
    if (weight_map_it == index.end() || !weight_map_it->is_object()) {
        add_evidence(result, index_path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size_or_zero(index_path));
        return false;
    }

    std::set<std::string> shard_names;
    for (const auto & item : weight_map_it->items()) {
        if (!item.value().is_string()) {
            add_evidence(result, index_path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size_or_zero(index_path));
            return false;
        }
        shard_names.insert(item.value().get<std::string>());
    }
    if (shard_names.empty()) {
        add_evidence(result, index_path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size_or_zero(index_path));
        return false;
    }

    add_evidence(result, index_path, COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_INDEX, file_size_or_zero(index_path));
    for (const auto & shard_name : shard_names) {
        candidates.push_back((index_path.parent_path() / fs::path(shard_name)).lexically_normal());
    }
    return true;
}

static void collect_directory(
        const fs::path & directory,
        std::vector<fs::path> & candidates,
        common_tree_draft_model_source & result) {
    add_evidence(result, directory, COMMON_TREE_DRAFT_MODEL_PROBE_DIRECTORY);

    std::vector<fs::path> indexes;
    std::error_code ec;
    for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec) || ec) {
            continue;
        }
        const std::string name = it->path().filename().generic_string();
        if (ends_with(name, ".safetensors.index.json")) {
            indexes.push_back(it->path());
        } else if (ends_with(name, ".gguf") || ends_with(name, ".safetensors")) {
            candidates.push_back(it->path());
        }
    }
    if (ec) {
        add_evidence(result, directory, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID);
        return;
    }

    std::sort(indexes.begin(), indexes.end());
    for (const auto & index : indexes) {
        collect_safetensors_index(index, candidates, result);
    }
}

static common_tree_draft_model_format probe_file(
        const fs::path & path,
        common_tree_draft_model_source & result) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_MISSING);
        return COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    }

    const uint64_t file_size = file_size_or_zero(path);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_MISSING, file_size);
        return COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    }

    std::array<unsigned char, 24> prefix{};
    const size_t prefix_size = static_cast<size_t>(std::min<uint64_t>(file_size, prefix.size()));
    if (prefix_size > 0 && !read_exact(file, prefix.data(), prefix_size)) {
        add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size, prefix_size);
        return COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    }

    if (prefix_size >= 24 &&
            prefix[0] == 'G' && prefix[1] == 'G' && prefix[2] == 'U' && prefix[3] == 'F') {
        add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_GGUF_HEADER, file_size, 24, 24);
        return COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF;
    }

    if (prefix_size < 8) {
        add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size, prefix_size);
        return COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    }

    std::array<unsigned char, 8> length_bytes{};
    std::copy_n(prefix.begin(), length_bytes.size(), length_bytes.begin());
    const uint64_t header_size = read_u64_le(length_bytes);
    if (header_size < 2 || header_size > SAFETENSORS_MAX_PROBE_HEADER || header_size > file_size - 8) {
        add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size, prefix_size, header_size);
        return COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    }

    std::string header(static_cast<size_t>(header_size), '\0');
    file.clear();
    file.seekg(8, std::ios::beg);
    if (!file || !read_exact(file, header.data(), header.size())) {
        add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size, 8, header_size);
        return COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    }

    try {
        const json object = json::parse(header);
        if (!object.is_object()) {
            add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size, 8 + header_size, header_size);
            return COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
        }
    } catch (const json::exception &) {
        add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_INVALID, file_size, 8 + header_size, header_size);
        return COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    }

    add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_HEADER, file_size, 8 + header_size, header_size);
    return COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS;
}

common_tree_draft_model_source common_tree_draft_model_source_probe(const std::string & input) {
    return common_tree_draft_model_source_probe(std::vector<std::string>{ input });
}

common_tree_draft_model_source common_tree_draft_model_source_probe(const std::vector<std::string> & inputs) {
    common_tree_draft_model_source result;
    std::vector<fs::path> candidates;

    for (const auto & input : inputs) {
        const fs::path path(input);
        std::error_code ec;
        if (fs::is_directory(path, ec) && !ec) {
            collect_directory(path, candidates, result);
        } else if (ends_with(path.filename().generic_string(), ".safetensors.index.json")) {
            collect_safetensors_index(path, candidates, result);
        } else {
            add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_FILE, file_size_or_zero(path));
            candidates.push_back(path);
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const fs::path & a, const fs::path & b) {
        return a.lexically_normal().generic_string() < b.lexically_normal().generic_string();
    });
    candidates.erase(std::unique(candidates.begin(), candidates.end(), [](const fs::path & a, const fs::path & b) {
        return a.lexically_normal().generic_string() == b.lexically_normal().generic_string();
    }), candidates.end());

    common_tree_draft_model_format selected = COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN;
    bool invalid = candidates.empty();
    for (const auto & path : candidates) {
        result.paths.push_back(path.lexically_normal().generic_string());
        const common_tree_draft_model_format format = probe_file(path, result);
        if (format == COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN) {
            invalid = true;
            continue;
        }
        if (selected == COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN) {
            selected = format;
        } else if (selected != format) {
            invalid = true;
            add_evidence(result, path, COMMON_TREE_DRAFT_MODEL_PROBE_AMBIGUOUS, file_size_or_zero(path));
        }
    }

    if (!invalid && selected != COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN) {
        result.format = selected;
        result.loader = selected == COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF
            ? COMMON_TREE_DRAFT_MODEL_LOADER_GGUF
            : COMMON_TREE_DRAFT_MODEL_LOADER_SAFETENSORS;
        result.confidence = 1.0f;
    }

    std::sort(result.evidence.begin(), result.evidence.end(), [](const auto & a, const auto & b) {
        if (a.path != b.path) {
            return a.path < b.path;
        }
        return a.signal < b.signal;
    });
    return result;
}

const char * common_tree_draft_model_format_name(common_tree_draft_model_format format) {
    switch (format) {
        case COMMON_TREE_DRAFT_MODEL_FORMAT_UNKNOWN:     return "unknown";
        case COMMON_TREE_DRAFT_MODEL_FORMAT_GGUF:        return "gguf";
        case COMMON_TREE_DRAFT_MODEL_FORMAT_SAFETENSORS: return "safetensors";
    }
    return "unknown";
}

const char * common_tree_draft_model_loader_name(common_tree_draft_model_loader loader) {
    switch (loader) {
        case COMMON_TREE_DRAFT_MODEL_LOADER_NONE:        return "none";
        case COMMON_TREE_DRAFT_MODEL_LOADER_GGUF:        return "gguf";
        case COMMON_TREE_DRAFT_MODEL_LOADER_SAFETENSORS: return "safetensors";
    }
    return "none";
}

const char * common_tree_draft_model_probe_signal_name(common_tree_draft_model_probe_signal signal) {
    switch (signal) {
        case COMMON_TREE_DRAFT_MODEL_PROBE_FILE:               return "file";
        case COMMON_TREE_DRAFT_MODEL_PROBE_DIRECTORY:          return "directory";
        case COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_INDEX:  return "safetensors_index";
        case COMMON_TREE_DRAFT_MODEL_PROBE_GGUF_HEADER:        return "gguf_header";
        case COMMON_TREE_DRAFT_MODEL_PROBE_SAFETENSORS_HEADER: return "safetensors_header";
        case COMMON_TREE_DRAFT_MODEL_PROBE_MISSING:            return "missing";
        case COMMON_TREE_DRAFT_MODEL_PROBE_INVALID:            return "invalid";
        case COMMON_TREE_DRAFT_MODEL_PROBE_AMBIGUOUS:          return "ambiguous";
    }
    return "invalid";
}
