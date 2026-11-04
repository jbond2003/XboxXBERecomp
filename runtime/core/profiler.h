#pragma once

// Perf diagnostic (xbe_perf / XBE_PERF): samples the registered threads about 2000 times a second and logs every five
// seconds where each spends its time, by guest function for the recompiled game code and by module and symbol
// otherwise. Windows only; elsewhere nothing.
namespace profiler {

// the calling thread is sampled from now on, under this name
void AddThisThread(const char* name);

}  // namespace profiler
