#include "tree-draft-tokenizer-metadata.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <sstream>

namespace {

bool parse_u64(const common_tree_draft_model_metadata_unknown & item, uint64_t & value) {
    if (!item.raw.empty() && item.raw.size() <= sizeof(value)) {
        if (item.gguf_type != GGUF_TYPE_UINT8 && item.gguf_type != GGUF_TYPE_UINT16 && item.gguf_type != GGUF_TYPE_UINT32 && item.gguf_type != GGUF_TYPE_UINT64) {
            return false;
        }
        value = 0;
        for (size_t i = 0; i < item.raw.size(); ++i) {
            value |= static_cast<uint64_t>(item.raw[i]) << (8 * i);
        }
        return true;
    }
    if (item.strings.size() != 1 || item.strings[0].empty() || item.strings[0].front() == '-') {
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long parsed = std::strtoull(item.strings[0].c_str(), &end, 10);
    if (errno == ERANGE || end == item.strings[0].c_str() || *end != '\0') {
        return false;
    }
    value = static_cast<uint64_t>(parsed);
    return true;
}

std::optional<std::string> single_string(const common_tree_draft_model_metadata_unknown & item) {
    if (item.strings.size() == 1) {
        return item.strings[0];
    }
    return std::nullopt;
}

void hash_byte(uint64_t & hash, unsigned char byte) {
    hash ^= byte;
    hash *= 1099511628211ULL;
}

void hash_u64(uint64_t & hash, uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) {
        hash_byte(hash, static_cast<unsigned char>((value >> (i * 8)) & 0xff));
    }
}

void hash_string(uint64_t & hash, const std::string & value) {
    hash_u64(hash, value.size());
    for (unsigned char byte : value) {
        hash_byte(hash, byte);
    }
}

std::string hash_strings(const std::vector<std::string> & values) {
    uint64_t hash = 14695981039346656037ULL;
    hash_u64(hash, values.size());
    for (const auto & value : values) {
        hash_string(hash, value);
    }
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(16) << hash;
    return stream.str();
}

bool is_special_token_key(const std::string & key, std::string & role) {
    static const std::vector<std::pair<std::string, std::string>> keys = {
        { "tokenizer.ggml.bos_token_id", "bos" },
        { "tokenizer.ggml.eos_token_id", "eos" },
        { "tokenizer.ggml.padding_token_id", "pad" },
        { "tokenizer.ggml.unknown_token_id", "unk" },
        { "tokenizer.ggml.separator_token_id", "sep" },
        { "bos_token_id", "bos" },
        { "eos_token_id", "eos" },
        { "pad_token_id", "pad" },
        { "unk_token_id", "unk" },
        { "sep_token_id", "sep" },
    };
    for (const auto & item : keys) {
        if (key == item.first) {
            role = item.second;
            return true;
        }
    }
    return false;
}

template<typename T>
void compare_optional(
        const char * name,
        const std::optional<T> & lhs,
        const std::optional<T> & rhs,
        std::vector<std::string> & mismatched,
        std::vector<std::string> & missing) {
    if (lhs && rhs) {
        if (*lhs != *rhs) {
            mismatched.emplace_back(name);
        }
    } else {
        missing.emplace_back(name);
    }
}

}

common_tree_draft_tokenizer_signature common_tree_draft_tokenizer_metadata_normalize(
        const common_tree_draft_model_metadata & metadata) {
    common_tree_draft_tokenizer_signature result;
    if (metadata.vocab_size) {
        result.vocab_size = metadata.vocab_size->value;
    }

    std::optional<std::vector<std::string>> tokens;
    std::optional<std::vector<std::string>> merges;
    std::optional<std::string> declared_vocab_hash;
    std::optional<std::string> declared_merge_hash;

    for (const auto & item : metadata.unknown) {
        const auto & key = item.origin.key;
        if (key == "tokenizer.ggml.model" || key == "tokenizer_family" || key == "tokenizer_class") {
            if (!result.tokenizer_family) {
                result.tokenizer_family = single_string(item);
            }
            continue;
        }
        if (key == "tokenizer.ggml.tokens" && !item.strings.empty()) {
            tokens = item.strings;
            continue;
        }
        if (key == "tokenizer.ggml.merges" && !item.strings.empty()) {
            merges = item.strings;
            continue;
        }
        if (key == "vocab_hash" || key == "tokenizer_vocab_hash") {
            declared_vocab_hash = single_string(item);
            continue;
        }
        if (key == "merges_hash" || key == "tokenizer_model_hash" || key == "merge_or_model_hash") {
            declared_merge_hash = single_string(item);
            continue;
        }

        std::string role;
        if (is_special_token_key(key, role)) {
            result.has_special_token_metadata = true;
            uint64_t id = 0;
            if (parse_u64(item, id)) {
                result.special_tokens.push_back({ std::move(role), id });
            } else {
                result.special_token_metadata_valid = false;
            }
        }
    }

    std::sort(result.special_tokens.begin(), result.special_tokens.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.role != rhs.role ? lhs.role < rhs.role : lhs.id < rhs.id;
    });
    for (size_t i = 1; i < result.special_tokens.size(); ++i) {
        if (result.special_tokens[i - 1].role == result.special_tokens[i].role && result.special_tokens[i - 1].id != result.special_tokens[i].id) {
            result.special_token_metadata_valid = false;
        }
    }
    result.special_tokens.erase(
            std::unique(result.special_tokens.begin(), result.special_tokens.end(), [](const auto & lhs, const auto & rhs) {
                return lhs.role == rhs.role && lhs.id == rhs.id;
            }),
            result.special_tokens.end());

    if (tokens) {
        result.vocab_hash = hash_strings(*tokens);
        if (!result.vocab_size) {
            result.vocab_size = tokens->size();
        }
    } else {
        result.vocab_hash = declared_vocab_hash;
    }
    if (merges) {
        result.merge_or_model_hash = hash_strings(*merges);
    } else {
        result.merge_or_model_hash = declared_merge_hash;
    }

    return result;
}

common_tree_draft_tokenizer_compatibility_result common_tree_draft_tokenizer_compare(
        const common_tree_draft_tokenizer_signature & lhs,
        const common_tree_draft_tokenizer_signature & rhs) {
    common_tree_draft_tokenizer_compatibility_result result;
    compare_optional("tokenizer_family", lhs.tokenizer_family, rhs.tokenizer_family, result.mismatched_fields, result.missing_fields);
    compare_optional("vocab_size", lhs.vocab_size, rhs.vocab_size, result.mismatched_fields, result.missing_fields);
    compare_optional("vocab_hash", lhs.vocab_hash, rhs.vocab_hash, result.mismatched_fields, result.missing_fields);
    compare_optional("merge_or_model_hash", lhs.merge_or_model_hash, rhs.merge_or_model_hash, result.mismatched_fields, result.missing_fields);

    if (lhs.has_special_token_metadata && rhs.has_special_token_metadata && lhs.special_token_metadata_valid && rhs.special_token_metadata_valid) {
        if (lhs.special_tokens.size() != rhs.special_tokens.size()) {
            result.mismatched_fields.emplace_back("special_token_map");
        } else {
            for (size_t i = 0; i < lhs.special_tokens.size(); ++i) {
                if (lhs.special_tokens[i].role != rhs.special_tokens[i].role || lhs.special_tokens[i].id != rhs.special_tokens[i].id) {
                    result.mismatched_fields.emplace_back("special_token_map");
                    break;
                }
            }
        }
    } else {
        result.missing_fields.emplace_back("special_token_map");
    }

    if (!result.mismatched_fields.empty()) {
        result.status = COMMON_TREE_DRAFT_TOKENIZER_INCOMPATIBLE;
    } else if (!result.missing_fields.empty()) {
        result.status = COMMON_TREE_DRAFT_TOKENIZER_PARTIAL;
    } else {
        result.status = COMMON_TREE_DRAFT_TOKENIZER_EXACT;
    }
    return result;
}
