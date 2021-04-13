// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>

namespace intel {
namespace hexl {

#undef TRUE   // MSVC defines TRUE
#undef FALSE  // MSVC defines FALSE

/// @enum CMPINT
/// @brief Represents binary operations between two boolean values
enum class CMPINT {
  EQ = 0,     ///< Equal
  LT = 1,     ///< Less than
  LE = 2,     ///< Less than or equal
  FALSE = 3,  ///< False
  NE = 4,     ///< Not equal
  NLT = 5,    ///< Not less than
  NLE = 6,    ///< Not less than or equal
  TRUE = 7    ///< True
};

/// @brief Returns the logical negation of a binary operation
/// @param[in] cmp The binary operation to negate
inline CMPINT Not(CMPINT cmp) {
  switch (cmp) {
    case CMPINT::EQ:
      return CMPINT::NE;
    case CMPINT::LT:
      return CMPINT::NLT;
    case CMPINT::LE:
      return CMPINT::NLE;
    case CMPINT::FALSE:
      return CMPINT::TRUE;
    case CMPINT::NE:
      return CMPINT::EQ;
    case CMPINT::NLT:
      return CMPINT::LT;
    case CMPINT::NLE:
      return CMPINT::LE;
    case CMPINT::TRUE:
      return CMPINT::FALSE;
    default:
      return CMPINT::FALSE;
  }
}

struct allocator_base {
  virtual ~allocator_base() noexcept {}
  virtual void* allocate(std::size_t bytes_count) = 0;
  virtual void deallocate(void* p, std::size_t n) = 0;
};

template <class AllocatorImpl>
struct allocator_interface : public allocator_base {
  // override interface & delegate implementation to AllocatorImpl
  void* allocate(std::size_t bytes_count) override {
    return static_cast<AllocatorImpl*>(this)->allocate_impl(bytes_count);
  }
  void deallocate(void* p, std::size_t n) override {
    static_cast<AllocatorImpl*>(this)->deallocate_impl(p, n);
  }

 private:
  // in case of AllocatorImpl doesn't provide implementations use default
  // behavior: break compilation with error
  void* allocate_impl(std::size_t bytes_count) {
    (void)bytes_count;
    fail_message<0>();
    return nullptr;
  }
  void deallocate_impl(void* p, std::size_t n) {
    (void)p;
    (void)n;
    fail_message<1>();
  }

  // pretty compilation error printing
  template <int error_code>
  static constexpr int fail_message() {
    static_assert(!(error_code == 0),
                  "Using 'allocator_adapter`as interface requires to implement "
                  "'::allocate_impl` method");
    static_assert(!(error_code == 1),
                  "Using 'allocator_adapter`as interface requires to implement "
                  "'::deallocate_impl` method");
    return 0;
  }
};
}  // namespace hexl
}  // namespace intel
