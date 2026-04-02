#include "download.h"

#include <string>

#undef NDEBUG
#include <cassert>

static void test_get_gguf_split_info() {
    printf("test_get_gguf_split_info\n");

    // non-gguf returns empty
    {
        auto s = get_gguf_split_info("model.bin");
        assert(s.prefix.empty());
    }

    // simple model with tag
    {
        auto s = get_gguf_split_info("model-Q4_K_M.gguf");
        assert(s.prefix == "model-Q4_K_M");
        assert(s.tag    == "Q4_K_M");
        assert(s.index  == 1);
        assert(s.count  == 1);
    }

    // tag with dot separator
    {
        auto s = get_gguf_split_info("model.F16.gguf");
        assert(s.prefix == "model.F16");
        assert(s.tag    == "F16");
        assert(s.index  == 1);
        assert(s.count  == 1);
    }

    // split file with tag
    {
        auto s = get_gguf_split_info("model-Q4_K_M-00002-of-00005.gguf");
        assert(s.prefix == "model-Q4_K_M");
        assert(s.tag    == "Q4_K_M");
        assert(s.index  == 2);
        assert(s.count  == 5);
    }

    // split file without tag
    {
        auto s = get_gguf_split_info("model-00001-of-00003.gguf");
        assert(s.prefix == "model");
        assert(s.tag.empty());
        assert(s.index  == 1);
        assert(s.count  == 3);
    }

    // no separator means no tag
    {
        auto s = get_gguf_split_info("weights.gguf");
        assert(s.prefix == "weights");
        assert(s.tag.empty());
        assert(s.index  == 1);
        assert(s.count  == 1);
    }

    // path with directory
    {
        auto s = get_gguf_split_info("models/repo/model-Q8_0.gguf");
        assert(s.prefix == "models/repo/model-Q8_0");
        assert(s.tag    == "Q8_0");
        assert(s.index  == 1);
        assert(s.count  == 1);
    }

    // lowercase tag gets uppercased
    {
        auto s = get_gguf_split_info("model-q4_0.gguf");
        assert(s.prefix == "model-q4_0");
        assert(s.tag    == "Q4_0");
    }

    // split with path and tag
    {
        auto s = get_gguf_split_info("dir/name-F16-00003-of-00010.gguf");
        assert(s.prefix == "dir/name-F16");
        assert(s.tag    == "F16");
        assert(s.index  == 3);
        assert(s.count  == 10);
    }

    printf("  All passed.\n");
}

static void test_common_download_split_repo_tag() {
    printf("test_common_download_split_repo_tag\n");

    {
        auto [repo, tag] = common_download_split_repo_tag("owner/model:Q4_K_M");
        assert(repo == "owner/model");
        assert(tag  == "Q4_K_M");
    }

    {
        auto [repo, tag] = common_download_split_repo_tag("owner/model");
        assert(repo == "owner/model");
        assert(tag.empty());
    }

    printf("  All passed.\n");
}

int main() {
    test_get_gguf_split_info();
    test_common_download_split_repo_tag();
    printf("\nAll download tests passed.\n");
    return 0;
}
