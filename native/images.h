#pragma once
#include "mtmd-helper.h"
#include "mtmd.h"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace winnow {
inline std::vector<unsigned char> image_bytes(const std::string &url) {
    auto comma = url.find(',');
    if (url.rfind("data:image/", 0) != 0 || comma == std::string::npos || comma < 7 ||
        url.substr(comma - 7, 7) != ";base64")
        throw std::invalid_argument(
            "Images must be base64 image data URLs, not remote URLs or filesystem paths");
    const auto s = url.substr(comma + 1);
    if (s.empty() || s.size() % 4 || s.size() > 24 * 1024 * 1024)
        throw std::invalid_argument("Invalid image payload size");
    std::vector<unsigned char> out;
    out.reserve(s.size() / 4 * 3);
    auto digit = [](char c) -> int {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        return -1;
    };
    for (size_t i = 0; i < s.size(); i += 4) {
        int a = digit(s[i]), b = digit(s[i + 1]);
        int c = s[i + 2] == '=' ? 0 : digit(s[i + 2]), d = s[i + 3] == '=' ? 0 : digit(s[i + 3]);
        bool pad2 = s[i + 2] == '=', pad1 = s[i + 3] == '=';
        if (a < 0 || b < 0 || c < 0 || d < 0 || (pad2 && !pad1) || ((pad1 || pad2) && i + 4 != s.size()) ||
            (pad2 && (b & 15)) || (!pad2 && pad1 && (c & 3)))
            throw std::invalid_argument("Invalid base64 image");
        out.push_back((a << 2) | (b >> 4));
        if (!pad2)
            out.push_back(((b & 15) << 4) | (c >> 2));
        if (!pad1)
            out.push_back(((c & 3) << 6) | d);
    }
    return out;
}
using Bitmap = std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)>;
using Chunks = std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)>;
struct Images {
    std::vector<Bitmap> owned;
    std::vector<const mtmd_bitmap *> pointers;
    std::vector<std::string> ids;
};
} // namespace winnow
