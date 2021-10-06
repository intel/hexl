// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <memory>
#include <vector>

#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/allocator.hpp"
#include "hexl/util/defines.hpp"

namespace intel {
namespace hexl {

/// @brief Allocater implementation using malloc and free
struct MallocStrategy : AllocatorBase {
  void* allocate(size_t bytes_count) final { return std::malloc(bytes_count); }

  void deallocate(void* p, size_t n) final {
    HEXL_UNUSED(n);
    std::free(p);
  }
};

using AllocatorStrategyPtr = std::shared_ptr<AllocatorBase>;
extern AllocatorStrategyPtr mallocStrategy;

/// @brief Allocates memory aligned to Alignment-byte sized boundaries
/// @details Alignment must be a power of two
template <typename T, uint64_t Alignment>
class AlignedAllocator {
 public:
  template <typename, uint64_t>
  friend class AlignedAllocator;

  using value_type = T;

  explicit AlignedAllocator(AllocatorStrategyPtr strategy = nullptr) noexcept
      : m_alloc_impl((strategy != nullptr) ? strategy : mallocStrategy) {}

  AlignedAllocator(const AlignedAllocator& src) = default;
  AlignedAllocator& operator=(const AlignedAllocator& src) = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Alignment>& src)
      : m_alloc_impl(src.m_alloc_impl) {}

  ~AlignedAllocator() {}

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Alignment>;
  };

  bool operator==(const AlignedAllocator&) { return true; }

  bool operator!=(const AlignedAllocator&) { return false; }

  /// @brief Allocates \p n elements aligned to Alignment-byte boundaries
  /// @return Pointer to the aligned allocated memory
  T* allocate(size_t n) {
    if (!IsPowerOfTwo(Alignment)) {
      return nullptr;
    }
    // Allocate enough space to ensure the alignment can be satisfied
    size_t buffer_size = sizeof(T) * n + Alignment;
    // Additionally, allocate a prefix to store the memory location of the
    // unaligned buffer
    size_t alloc_size = buffer_size + sizeof(void*);
    void* buffer = m_alloc_impl->allocate(alloc_size);
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

  void deallocate(T* p, size_t n) {
    if (!p) {
      return;
    }
    void* store_buffer_addr = (reinterpret_cast<char*>(p) - sizeof(void*));
    void* free_address = *(static_cast<void**>(store_buffer_addr));
    m_alloc_impl->deallocate(free_address, n);
  }

 private:
  AllocatorStrategyPtr m_alloc_impl;
};

/// @brief 64-byte aligned memory allocator
template <typename T>
using AlignedVector64 = std::vector<T, AlignedAllocator<T, 64> >;

}  // namespace hexl
}  // namespace intel
