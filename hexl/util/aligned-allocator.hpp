// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "number-theory/number-theory.hpp"

namespace intel {
namespace hexl {

template <typename T, uint64_t Alignment>
class AlignedAllocator {
 public:
  using value_type = T;

  AlignedAllocator() noexcept {}

  AlignedAllocator(const AlignedAllocator&) {}

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

  ~AlignedAllocator() {}

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  bool operator==(const AlignedAllocator&) { return true; }

  bool operator!=(const AlignedAllocator&) { return false; }

  T* allocate(std::size_t n) {
    if (!IsPowerOfTwo(Alignment)) {
      return nullptr;
    }
    // Allocate enough space to ensure the alignment can be satisfied
    size_t buffer_size = sizeof(T) * n + Alignment;
    // Additionally, allocate a prefix to store the memory location of the
    // unaligned buffer
    size_t alloc_size = buffer_size + sizeof(void*);
    void* buffer = std::malloc(alloc_size);
    if (!buffer) {
      return nullptr;
    }

    // Reserve first location for pointer to originally-allocated space
    void* aligned_buffer = static_cast<char*>(buffer) + sizeof(void*);
    std::align(Alignment, sizeof(T) * n, aligned_buffer, buffer_size);
    if (!aligned_buffer) {
      return nullptr;
    }

    // Store allocated buffer address at aligned_buffer - sizeof(void*).
    void* store_buffer_addr =
        static_cast<char*>(aligned_buffer) - sizeof(void*);
    *(static_cast<void**>(store_buffer_addr)) = buffer;

    return static_cast<T*>(aligned_buffer);
  }

  void deallocate(T* p, std::size_t n) {
    if (!p) {
      return;
    }
    void* store_buffer_addr = (reinterpret_cast<char*>(p) - sizeof(void*));
    void* free_address = *(static_cast<void**>(store_buffer_addr));
    (void)n;  // Avoid unused variable
    std::free(free_address);
  }
};

template <typename T>
using AlignedVector64 = std::vector<T, AlignedAllocator<T, 64> >;

}  // namespace hexl
}  // namespace intel
