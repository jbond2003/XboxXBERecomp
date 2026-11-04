#pragma once
#include <mutex>

#include "gpu/state.h"

namespace d3d {

gpu::State& State();
std::recursive_mutex& Lock();
// the title's thread: presents show the last picture drawn until the title draws again (gpu::Backend::HoldFrame)
void HoldFrame();

}  // namespace d3d
