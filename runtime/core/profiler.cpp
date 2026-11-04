#include "core/profiler.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/diag.h"
#include "core/log.h"

#pragma comment(lib, "dbghelp.lib")

namespace dispatch {
uint32_t GuestAt(uintptr_t pc);  // cpu/dispatch.cpp
}

namespace profiler {

namespace {

constexpr auto kReport = std::chrono::seconds(5);
constexpr size_t kTop = 14;

struct Sampled {
    std::string name;
    HANDLE thread = nullptr;
    std::unordered_map<uintptr_t, uint32_t> pcs;  // sample counts by host pc
    uint64_t samples = 0;
};

std::mutex g_lock;
std::vector<Sampled> g_threads;

// a pc's module and its base
std::string ModuleOf(uintptr_t pc, uintptr_t& base) {
    HMODULE m = nullptr;
    base = 0;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(pc), &m) || !m)
        return "?";
    base = reinterpret_cast<uintptr_t>(m);
    char path[MAX_PATH] = {};
    GetModuleFileNameA(m, path, MAX_PATH);
    const char* slash = std::max(std::strrchr(path, '\\'), std::strrchr(path, '/'));
    return slash ? slash + 1 : path;
}

// what a sample is counted as: the guest function, else module!symbol, else module+offset by 4 KB
std::string Bucket(uintptr_t pc) {
    if (uint32_t guest = dispatch::GuestAt(pc)) {
        char b[32];
        std::snprintf(b, sizeof b, "game sub_%08X", guest);
        return b;
    }
    uintptr_t base;
    std::string module = ModuleOf(pc, base);
    alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256] = {};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buffer);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 displacement = 0;
    if (SymFromAddr(GetCurrentProcess(), pc, &displacement, sym)) return module + "!" + sym->Name;
    char b[48];
    std::snprintf(b, sizeof b, "+%llX", (unsigned long long)((pc - base) & ~uintptr_t(0xFFF)));
    return module + b;
}

void Report() {
    std::lock_guard lock(g_lock);
    for (Sampled& t : g_threads) {
        if (!t.samples) continue;
        std::unordered_map<std::string, uint64_t> buckets;
        for (const auto& [pc, n] : t.pcs) buckets[Bucket(pc)] += n;
        std::vector<std::pair<uint64_t, std::string>> top;
        for (auto& [name, n] : buckets) top.push_back({n, name});
        std::sort(top.rbegin(), top.rend());
        std::string line;
        for (size_t i = 0; i < top.size() && i < kTop; i++) {
            char b[160];
            std::snprintf(b, sizeof b, "%s%.1f%% %s", i ? ", " : "", 100.0 * double(top[i].first) / double(t.samples),
                          top[i].second.c_str());
            line += b;
        }
        LOGR("profile: %s thread (%llu samples): %s", t.name.c_str(), (unsigned long long)t.samples, line.c_str());
        t.pcs.clear();
        t.samples = 0;
    }
}

void Sample() {
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    auto last = std::chrono::steady_clock::now();
    for (;;) {
        if (timer) {
            LARGE_INTEGER due;
            due.QuadPart = -5000;  // 0.5 ms
            SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE);
            WaitForSingleObject(timer, INFINITE);
        } else {
            Sleep(1);
        }
        {
            std::lock_guard lock(g_lock);
            for (Sampled& t : g_threads) {
                if (SuspendThread(t.thread) == DWORD(-1)) continue;
                CONTEXT c{};
                c.ContextFlags = CONTEXT_CONTROL;
                bool got = GetThreadContext(t.thread, &c);
                ResumeThread(t.thread);  // before anything that allocates: the thread may hold the heap's lock
                if (got) {
                    t.pcs[uintptr_t(c.Rip)]++;
                    t.samples++;
                }
            }
        }
        auto now = std::chrono::steady_clock::now();
        if (now - last >= kReport) {
            Report();
            last = now;
        }
    }
}

}  // namespace

void AddThisThread(const char* name) {
    if (!diag::Perf()) return;
    HANDLE h = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &h,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, 0))
        return;
    static std::once_flag started;
    {
        std::lock_guard lock(g_lock);
        g_threads.push_back({name, h});
    }
    std::call_once(started, [] { std::thread(Sample).detach(); });
    LOGR("profile: sampling the %s thread", name);
}

}  // namespace profiler

#else

namespace profiler {
void AddThisThread(const char*) {}
}  // namespace profiler

#endif
