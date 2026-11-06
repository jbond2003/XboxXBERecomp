#pragma once
#include <cstddef>

// Host stacks a game hook can switch between on one thread. Some titles run script threads as coroutines that copy
// their part of the guest stack out and back and jump into the middle of a function; recompiled code keeps frames on
// the host stack too, so such a coroutine needs a host stack of its own while it is suspended.
namespace fiber {

struct Fiber;
using Entry = void (*)(void* arg);

// the calling thread as a fiber, made one on first use
Fiber* Current();
// a fiber that runs entry(arg) when first switched to; entry must never return, it switches away for good instead
Fiber* Create(Entry entry, void* arg, size_t stack_size = 4u << 20);
void Switch(Fiber* to);
// never the running fiber
void Destroy(Fiber* f);

}  // namespace fiber
