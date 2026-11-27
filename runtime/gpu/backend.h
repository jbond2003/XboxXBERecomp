#pragma once
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>

#include "gpu/state.h"

namespace gpu {

// D3DCLEAR flags as the XDK defines them
enum : uint32_t {
    kClearTarget = 0xF0,  // ARGB channels
    kClearZ = 0x01,
    kClearStencil = 0x02,
    // Not a D3DCLEAR value: ambient occlusion over what the target holds now, in order with the draws. The rects are
    // kOcclusionRects of floats in place of ints, as d3d::Occlusion (xdk/d3d8.h) fills them
    kClearOcclusion = 0x100,
    // Not a D3DCLEAR value: this frame's shadow picture is read by its depth (d3d::SunShadows), kSunRects of floats
    kClearSun = 0x200,
    // Not a D3DCLEAR value: where the next d3d::Occlusion's view lies in the shadow picture (d3d::SunOcclusion),
    // kSunViewRects of floats
    kClearSunView = 0x400,
};
constexpr uint32_t kOcclusionRects = 4;
constexpr uint32_t kSunRects = 2;
constexpr uint32_t kSunViewRects = 4;

class Backend {
public:
    virtual ~Backend() = default;
    // the title's back buffer size, the output window may differ
    virtual bool Init(uint32_t width, uint32_t height) = 0;
    // rects: count D3DRECTs (4 x int32), null for the whole viewport
    virtual void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
                       uint32_t stencil) = 0;
    // indices: the title's 16 bit indices, null for non indexed draws, then start is the first vertex
    virtual void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) = 0;
    virtual void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t src_rects_va,
                      uint32_t dst_points_va) = 0;
    virtual void Present(const Surface& front) = 0;
    // until the title draws again, presents show the last picture drawn: a stopped movie's frame buffers each hold
    // another of its pictures, and presents with nothing drawn would flick between them
    virtual void HoldFrame() {}
    // the CPU is about to touch this resource's memory: write GPU results back first
    virtual void Sync(uint32_t data_va, uint32_t size) = 0;
    // a lock that does not wait for the GPU (NOOVERWRITE): a result a frame or two old is what the console gives too
    virtual void SyncLate(uint32_t data_va) = 0;
    // SyncLate in two for the render thread: DeliverLate (any thread) writes the newest finished copy to guest memory,
    // false when there is none recent; RecordLate (in order with the draws) starts a copy for a later lock
    virtual bool DeliverLate(uint32_t) { return false; }
    virtual void RecordLate(uint32_t) {}
    virtual void Flush() = 0;
    virtual void SetGamma(const uint8_t* ramp /*3 x 256*/) = 0;
    // visibility tests: draws count passing pixels while simple[kMCountEnable / 4] is set, ClearReport zeroes the
    // count, GetReport writes it to a 16 byte report once the GPU is past those draws; without them it stays pending
    virtual void ClearReport() {}
    virtual void GetReport(uint32_t /*report*/) {}

    // the XDK's fences and busy checks against the render thread; a backend that draws at once is never behind
    virtual uint32_t Fence() { return ++fence_; }
    virtual bool FencePending(uint32_t) { return false; }
    virtual void WaitFence(uint32_t) {}
    virtual bool Busy(uint32_t /*resource*/) { return false; }
    virtual void WaitResource(uint32_t /*resource*/) {}
    // until the GPU has drawn present number n, counted from 0; any thread
    virtual void WaitGpuFrame(uint64_t /*n*/) {}

private:
    uint32_t fence_ = 0;
};

std::unique_ptr<Backend> CreateBackend();
// the same backend driven from a thread of its own: calls return once recorded
std::unique_ptr<Backend> CreateThreadedBackend(std::unique_ptr<Backend> inner);

// perf diagnostic: the title's thread recording for the render thread, and waiting on it
inline std::atomic<uint64_t> g_perf_capture_ns{0}, g_perf_wait_ns{0}, g_perf_waits{0};
inline std::atomic<uint64_t> g_perf_wait_kind_ns[6];  // read-back, resource, fence, flush, copy, next frame
inline std::atomic<uint64_t> g_perf_sync_late{0}, g_perf_sync_block{0};  // read-backs the title waited for, by cause

// Depth of field around the sight picture: 0 none, up to 1 for all of it; a title sets it each frame from how far the
// player aims down the sights. Only for a title that draws the gun in a scene of its own after a clear of depth alone.
inline std::atomic<float> g_dof{0.0f};
// The same for the gun alone: the part of it nearest the eye out of focus, by its own depth.
inline std::atomic<float> g_gun_blur{0.0f};
// with g_gun_blur: where the blur ends for the gun in hand, in near planes of the gun's scene (0: the usual 30)
inline std::atomic<float> g_gun_reach{0.0f};
// A title holds SMAA off while this is set: its menus' text, drawn sharp, is softened by it
inline std::atomic<bool> g_smaa_off{false};
// A square render target of this console size (0: none) is made this many times larger still than the render scale
// makes it, up to 4096 host pixels: a title's shadow picture laid over more of the world than it was made for
inline std::atomic<uint32_t> g_boost_size{0};
inline std::atomic<float> g_boost{1.0f};
// Set by the renderer while the shadow picture keeps its depth as d3d::SunShadows asks (a depth target made beside
// it, its shaders compiled): a title puts in the picture what only a depth test can tell apart from what it falls on
inline std::atomic<bool> g_sun_depth_ok{false};

// F3: save the next frame, every draw and target, under diag::CaptureDir
inline std::atomic<bool> g_capture{false};

// A title asks for a picture of the next frame presented, as a 32 bit .bmp: the rows from top to bottom of the frame
// (fractions of its height), `width` pixels wide, the frame taken as a 4:3 picture. For one made from the player's own
// game, as a boot splash of the title's own first screen.
struct FrameSave {
    std::filesystem::path path;
    float top = 0.0f, bottom = 1.0f;
    uint32_t width = 640;
};
inline std::mutex g_frame_save_lock;
inline std::optional<FrameSave> g_frame_save;
inline void SaveFrame(FrameSave save) {
    std::lock_guard lock(g_frame_save_lock);
    g_frame_save = std::move(save);
}
inline std::optional<FrameSave> TakeFrameSave() {
    std::lock_guard lock(g_frame_save_lock);
    std::optional<FrameSave> save = std::move(g_frame_save);
    g_frame_save.reset();
    return save;
}

}  // namespace gpu
