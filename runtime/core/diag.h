#pragma once
#include <cstdlib>
#include <filesystem>
#include <iterator>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// diagnostics switched on by environment variables (or a file, for perf), read once
namespace diag {

// XBE_PERF set, or a file named xbe_perf (any extension) beside the exe: perf log lines once a second and the
// read-backs named
inline bool Perf() {
    static const bool on = [] {
        if (std::getenv("XBE_PERF")) return true;
#ifdef _WIN32
        wchar_t exe[1024];
        DWORD n = GetModuleFileNameW(nullptr, exe, DWORD(std::size(exe)));
        if (!n || n >= std::size(exe)) return false;
        std::error_code ec;
        for (const auto& e : std::filesystem::directory_iterator(std::filesystem::path(exe).parent_path(), ec))
            if (e.path().stem() == L"xbe_perf") return true;
#endif
        return false;
    }();
    return on;
}

// XBE_CAPTURE_DIR: where F3 saves a frame, empty: F3 does not capture
inline const std::filesystem::path& CaptureDir() {
    static const std::filesystem::path dir = [] {
#ifdef _WIN32
        const wchar_t* v = _wgetenv(L"XBE_CAPTURE_DIR");
#else
        const char* v = std::getenv("XBE_CAPTURE_DIR");
#endif
        return v ? std::filesystem::path(v) : std::filesystem::path();
    }();
    return dir;
}

}  // namespace diag
