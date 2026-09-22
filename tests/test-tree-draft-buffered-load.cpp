#include "tree-draft-buffered-load.h"

#include <cassert>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

int main() {
    const fs::path path = fs::temp_directory_path() / "tree-draft-buffered-load.bin";
    {
        std::ofstream f(path, std::ios::binary);
        for (int i = 0; i < 64; ++i) {
            const unsigned char v = static_cast<unsigned char>(i);
            f.write(reinterpret_cast<const char *>(&v), 1);
        }
    }
    common_tree_draft_model_source source;
    source.paths.push_back(path.generic_string());
    const common_tree_draft_buffer_interval intervals[] = {
        {0, 16, 16, 32},
        {0, 0, 8, 16},
    };
    std::vector<common_tree_draft_buffered_payload> payloads;
    assert(common_tree_draft_buffered_load(source, intervals, 2, COMMON_TREE_DRAFT_BUFFER_TRIGGER_CAPABILITY, &payloads) == COMMON_TREE_DRAFT_BUFFERED_LOAD_OK);
    assert(payloads.size() == 2);
    assert(payloads[0].source_offset == 0 && payloads[0].view.data[0] == 0 && payloads[0].view.data[7] == 7);
    assert(payloads[1].source_offset == 16 && payloads[1].view.data[0] == 16 && payloads[1].view.data[15] == 31);
    assert(reinterpret_cast<uintptr_t>(payloads[0].view.data) % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(payloads[1].view.data) % 32 == 0);
    assert(common_tree_draft_buffered_load(source, intervals, 2, COMMON_TREE_DRAFT_BUFFER_TRIGGER_CORRUPT, &payloads) == COMMON_TREE_DRAFT_BUFFERED_LOAD_TRIGGER);
    const common_tree_draft_buffer_interval bad[] = {{0, 60, 8, 16}};
    assert(common_tree_draft_buffered_load(source, bad, 1, COMMON_TREE_DRAFT_BUFFER_TRIGGER_POLICY, &payloads) == COMMON_TREE_DRAFT_BUFFERED_LOAD_RANGE);
    fs::remove(path);
    return 0;
}

