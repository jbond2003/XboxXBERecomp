#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "gpu/texture.h"

// A title's own pictures drawn in place of some of its textures, made by the title's code while it runs (keycaps in
// place of controller button icons, say). The renderers ask the provider once for each texture's bytes, and again for
// every texture after Changed; a picture may be any size, the texture's coordinates stretch over it.
namespace gpu::texture_override {

struct Picture {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint32_t> argb;  // 0xAARRGGBB, rows top down, straight alpha
};

// the picture for a 2D texture's bytes (level 0 at guest), null to draw the texture itself. render_scale is the
// output's size over the console's, for a picture that wants to be drawn sharp. Called on the render thread; the same
// pointer back is drawn from the GPU copy already made of it.
using Provider = std::shared_ptr<const Picture> (*)(const TextureDesc& d, const uint8_t* guest, float render_scale);

void SetProvider(Provider p);  // before boot
std::shared_ptr<const Picture> Find(const TextureDesc& d, const uint8_t* guest, float render_scale);
void Changed();      // any thread: what the provider gives has changed, every texture is asked again
uint32_t Version();  // moves with Changed

}  // namespace gpu::texture_override
