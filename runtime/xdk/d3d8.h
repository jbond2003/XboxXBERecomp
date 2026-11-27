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
// radius: how far a surface is shaded from, in the title's world units; strength: 1 as worked out, 2 twice as dark;
// fog: the fog already in the picture over view depth, min(saturate(fog[0] * z + fog[1]), fog[2]), and its colour
// (red, green, blue, 0 to 1): what the fog covers is not darkened. Null for no fog
void Occlusion(const float* m, bool gtao, float radius, float strength, const float* fog, const float* fog_colour);
// the title's thread, before this frame's shadow picture (the target of g_boost_size) is drawn: everything drawn into
// it keeps its depth along the picture's camera, and where the world lays the picture on itself it compares that depth
// instead of reading the picture's alpha, so the picture can hold the ground and what stands on it. From the stage's
// coordinates (u, v, s) over their w: depth = from . (u, v, s, 1), 0 at the picture's near plane and 1 at its far one,
// as the picture's own draws store it. gain: the shadow handed to the title's combiners in place of the alpha is
// times this; shade: where the stage before the picture's holds baked light, the shadow darkens it only to this level
// (0: not); bias: a receiver is lit up to this much depth behind what the picture holds; slope: the most depth per
// unit of u or v a receiver's plane is taken to have. gpu::g_sun_depth_ok says when it works.
void SunShadows(const float from[4], float gain, float shade, float bias, float slope);
// the title's thread, after SunShadows and before this frame's Occlusion: where the scene's view space lies in the
// shadow picture, three rows of four taking (x, y, z, 1) of view space to the picture's u, v and depth (as
// SunShadows' from gives it). In the sun the shade takes away only keep of what it does out of it (the ambient light it
// darkens is a small part of a sunlit surface's), so it matches the shadows the sun casts. bias: the depth a point
// may lie behind what the picture holds and still be lit.
void SunOcclusion(const float view_to_sun[12], float keep, float bias);

}  // namespace d3d
