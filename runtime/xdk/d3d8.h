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
//   kSunStill   only what does not move, or a share of it: added to the part the renderer is making under the
//               number id (begun empty when the number is new). This frame's world still reads the picture as the
//               frame before left it, whose camera may have been another: across, up and from give that picture's
//               u, v and depth from this frame's stage coordinates
//   kSunMoving  only what moves, drawn over a copy of the part number id: the one kept so far, or the one being
//               made, which then is the kept one. Where the renderer holds neither under that number
//               (gpu::g_sun_static_lost counts up) the frame's picture has what moves alone
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

// the title's thread, once a frame before the scene's world is drawn: surfaces drawn after it with a texture the
// title marked as mirroring on stage 0 (gpu::texture_override::kMirror; level ones, still water) are drawn a second
// time with what the last frame's picture held along each pixel's mirrored line of sight, rippled. Nothing else of
// the surface's draw changes, and a surface drawn blended is left as it is.
// m: the scene's camera to screen matrix, as Occlusion takes it; view: the world to camera matrix, 16 floats, row
// vectors (a rotation and a place, the world's up its third row); strength: 1 as worked out, less for less of the
// mirror; seconds: a clock for the ripples, counted round 64.
// The first frame, and one whose view the last frame's picture cannot serve, is drawn without.
// water: those surfaces are water drawn whole instead. The second draw covers the title's own with a water of the
// renderer's: waves that move, its own colour darker where it is looked down into, the mirror, the sky's colour where
// the picture has nothing to mirror, and the sun's glint. Far off (from 5000 units, all of it at 14000) and in the
// fog the title's surface is left to show. A frame without the last one's picture is drawn with the sky's colour.
constexpr int kWaterSplashes = 12;
struct Water {
    float tone[3];   // the water's own colour: what the title's picture of it averages
    float rough;     // how rough: 1 a sea, less for stiller water (above 0)
    float sky[3];    // the sky's colour
    float glint;     // how bright the sun glints in it, 0 for none
    float sun[3];    // the way to the sun in the world, a unit vector
    // What fell in it of late. The water is pushed down there, waves run out, it is churned white and the spray
    // speckles it as it falls back: how wide and for how long by the kind. x, y and z in the world (drawn on a water
    // within 40 units of that height); w 0 for none, else the seconds since it fell (under 16) + 16 times its kind:
    // 1 a rifle's round, 2 a heavy machine gun's, 3 a cannon's, 4 a small thing fallen in, 5 a large one, 6 a
    // grenade, 7 a tank's shell, 8 an artillery shell or a bomb.
    float splash[kWaterSplashes][4];
};
void Reflections(const float* m, const float* view, float strength, float seconds, const Water* water = nullptr);

}  // namespace d3d
