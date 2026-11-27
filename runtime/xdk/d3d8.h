#pragma once
#include <mutex>

#include "gpu/backend.h"
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
//
// The picture's depth is a target of the renderer's own, and what is drawn into the picture is drawn into that only,
// with no colour. A title whose picture holds much that does not move (a level's meshes) need not draw that every
// frame, while the picture's camera stays where it is: SunPicture::keep says what this frame's draws into it are.
//   kSunWhole   all of it, as a title that keeps nothing draws it
//   kSunStill   only what does not move, kept by the renderer as part number id. This frame's world still reads the
//               picture as the frame before left it, whose camera may have been another: across, up and from give
//               that picture's u, v and depth from this frame's stage coordinates
//   kSunMoving  only what moves, drawn over a copy of the kept part number id. Where the renderer no longer holds
//               that part (gpu::g_sun_static_lost counts up) the frame's picture has what moves alone
// In kSunWhole and kSunMoving across and up are {1, 0, 0, 0} and {0, 1, 0, 0}: the stage's own u and v.
// fade: the picture's shadows end, fading over a twentieth of it, fade[2] inside the square that lies fade[0], fade[1]
// from the picture's own in u and v: a kept picture's camera lags the place the title would put it, and the square
// the shadows end at need not jump with it.
using gpu::SunKeep, gpu::kSunWhole, gpu::kSunMoving, gpu::kSunStill;
struct SunPicture {
    float across[4] = {1, 0, 0, 0}, up[4] = {0, 1, 0, 0}, from[4] = {};
    float gain = 1, shade = 0, bias = 0, slope = 0;
    float fade[3] = {};
    SunKeep keep = kSunWhole;
    int id = 0;
};
void SunShadows(const SunPicture& p);
// the title's thread, after SunShadows and before this frame's Occlusion: where the scene's view space lies in the
// shadow picture, three rows of four taking (x, y, z, 1) of view space to the picture's u, v and depth (as
// SunShadows' from gives it). In the sun the shade takes away only keep of what it does out of it (the ambient light it
// darkens is a small part of a sunlit surface's), so it matches the shadows the sun casts. bias: the depth a point
// may lie behind what the picture holds and still be lit.
void SunOcclusion(const float view_to_sun[12], float keep, float bias);

}  // namespace d3d
