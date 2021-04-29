// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace intel {
namespace hexl {

struct AllocatorBase {
  virtual ~AllocatorBase() noexcept {}
  virtual void* allocate(size_t bytes_count) = 0;
  virtual void deallocate(void* p, size_t n) = 0;
};

template <class AllocatorImpl>
struct AllocatorInterface : public AllocatorBase {
  // override interface & delegate implementation to AllocatorImpl
  void* allocate(size_t bytes_count) override {
    return static_cast<AllocatorImpl*>(this)->allocate_impl(bytes_count);
  }
  void deallocate(void* p, size_t n) override {
    static_cast<AllocatorImpl*>(this)->deallocate_impl(p, n);
  }

 private:
  // in case of AllocatorImpl doesn't provide implementations use default
  // behavior: break compilation with error
  void* allocate_impl(size_t bytes_count) {
    (void)bytes_count;
    fail_message<0>();
    return nullptr;
  }
  void deallocate_impl(void* p, size_t n) {
    (void)p;
    (void)n;
    fail_message<1>();
  }

  // pretty compilation error printing
  template <int error_code>
  static constexpr int fail_message() {
    static_assert(!(error_code == 0),
                  "Using 'AllocatorAdapter`as interface requires to implement "
                  "'::allocate_impl` method");
    static_assert(!(error_code == 1),
                  "Using 'AllocatorAdapter`as interface requires to implement "
                  "'::deallocate_impl` method");
    return 0;
  }
};
}  // namespace hexl
}  // namespace intel
