#include "gpu/texture_override.h"

#include <atomic>

namespace gpu::texture_override {

namespace {
std::atomic<Provider> g_provider{nullptr};
std::atomic<uint32_t> g_version{1};
}  // namespace

void SetProvider(Provider p) {
    g_provider = p;
    Changed();
}

std::shared_ptr<const Picture> Find(const TextureDesc& d, const uint8_t* guest, float render_scale) {
    Provider p = g_provider.load();
    if (!p || !guest || d.cube || d.depth > 1 || d.linear) return nullptr;
    auto pic = p(d, guest, render_scale);
    if (pic && (!pic->width || !pic->height || pic->argb.size() != size_t(pic->width) * pic->height)) return nullptr;
    return pic;
}

void Changed() { g_version.fetch_add(1); }

uint32_t Version() { return g_version.load(); }

}  // namespace gpu::texture_override
