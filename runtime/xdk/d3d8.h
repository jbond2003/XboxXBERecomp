#pragma once
#include <mutex>

#include "gpu/state.h"

namespace d3d {

gpu::State& State();
std::recursive_mutex& Lock();
// the title's thread: presents show the last picture drawn until the title draws again (gpu::Backend::HoldFrame)
void HoldFrame();
// the title's thread, between its draws: darkens the corners and creases of what the render target holds now, from
// the depth drawn so far (a title calls it once its solid world is drawn and before smoke, glass and the like).
// m: the scene's camera to screen matrix, row vectors, as SetTransform takes one; gtao: GTAO, else SSAO;
// radius: how far a surface is shaded from, in the title's world units;
// fog: the fog already in the picture over view depth, min(saturate(fog[0] * z + fog[1]), fog[2]), and its colour
// (red, green, blue, 0 to 1): what the fog covers is not darkened. Null for no fog
void Occlusion(const float* m, bool gtao, float radius, const float* fog, const float* fog_colour);

}  // namespace d3d
