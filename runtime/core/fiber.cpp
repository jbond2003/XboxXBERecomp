#include "core/fiber.h"

#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <ucontext.h>
#endif

namespace fiber {

struct Fiber {
    Entry entry = nullptr;
    void* arg = nullptr;
    bool thread = false;  // the thread's own stack, not ours to free
#ifdef _WIN32
    void* handle = nullptr;
#else
    ucontext_t uc{};
    void* stack = nullptr;
    size_t stack_size = 0;
#endif
};

static thread_local Fiber* t_current = nullptr;

Fiber* Current() {
    if (t_current) return t_current;
    Fiber* f = new Fiber;
    f->thread = true;
#ifdef _WIN32
    f->handle = IsThreadAFiber() ? GetCurrentFiber() : ConvertThreadToFiberEx(nullptr, FIBER_FLAG_FLOAT_SWITCH);
    if (!f->handle) std::abort();
#endif
    t_current = f;
    return f;
}

#ifdef _WIN32
static void WINAPI Start(void* p) {
    Fiber* f = static_cast<Fiber*>(p);
    f->entry(f->arg);
    std::abort();  // entry returned
}
#else
static void Start(unsigned lo, unsigned hi) {
    Fiber* f = reinterpret_cast<Fiber*>((uintptr_t(hi) << 32) | uintptr_t(lo));
    f->entry(f->arg);
    std::abort();
}
#endif

Fiber* Create(Entry entry, void* arg, size_t stack_size) {
    Current();
    Fiber* f = new Fiber;
    f->entry = entry;
    f->arg = arg;
#ifdef _WIN32
    // reserved, committed as it grows
    f->handle = CreateFiberEx(64u << 10, stack_size, FIBER_FLAG_FLOAT_SWITCH, Start, f);
    if (!f->handle) {
        delete f;
        return nullptr;
    }
#else
    f->stack = mmap(nullptr, stack_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (f->stack == MAP_FAILED) {
        delete f;
        return nullptr;
    }
    f->stack_size = stack_size;
    getcontext(&f->uc);
    f->uc.uc_stack.ss_sp = f->stack;
    f->uc.uc_stack.ss_size = stack_size;
    f->uc.uc_link = nullptr;
    uintptr_t p = reinterpret_cast<uintptr_t>(f);
    makecontext(&f->uc, reinterpret_cast<void (*)()>(Start), 2, unsigned(p), unsigned(uint64_t(p) >> 32));
#endif
    return f;
}

void Switch(Fiber* to) {
    Fiber* from = Current();
    if (to == from) return;
    t_current = to;
#ifdef _WIN32
    SwitchToFiber(to->handle);
#else
    swapcontext(&from->uc, &to->uc);
#endif
}

void Destroy(Fiber* f) {
    if (!f || f == t_current || f->thread) return;
#ifdef _WIN32
    DeleteFiber(f->handle);
#else
    munmap(f->stack, f->stack_size);
#endif
    delete f;
}

}  // namespace fiber
