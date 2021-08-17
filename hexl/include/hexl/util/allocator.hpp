// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace intel {
namespace hexl {

/// @brief Base class for custom memory allocator
struct AllocatorBase {
  virtual ~AllocatorBase() noexcept {}

  /// @brief Allocates byte_count bytes of memory
  /// @param[in] bytes_count Number of bytes to allocate
  /// @return A pointer to the allocated memory
  virtual void* allocate(size_t bytes_count) = 0;

  /// @brief Deallocate memory
  /// @param[in] p Pointer to memory to deallocate
  /// @param[in] n Number of bytes to deallocate
  virtual void deallocate(void* p, size_t n) = 0;
};

/// @brief Helper memory allocation struct which delegates implementation to
/// AllocatorImpl
template <class AllocatorImpl>
struct AllocatorInterface : public AllocatorBase {
  /// @brief Override interface and delegate implementation to AllocatorImpl
  void* allocate(size_t bytes_count) override {
    return static_cast<AllocatorImpl*>(this)->allocate_impl(bytes_count);
  }

  /// @brief Override interface and delegate implementation to AllocatorImpl
  void deallocate(void* p, size_t n) override {
    static_cast<AllocatorImpl*>(this)->deallocate_impl(p, n);
  }

 private:
  // in case AllocatorImpl doesn't provide implementations, use default null
  // behavior
  void* allocate_impl(size_t bytes_count) {
    HEXL_UNUSED(bytes_count);
    return nullptr;
  }
  void deallocate_impl(void* p, size_t n) {
    HEXL_UNUSED(p);
    HEXL_UNUSED(n);
  }
};
}  // namespace hexl
}  // namespace intel
