// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cmath>

#include "hexl/util/check.hpp"
#include "hexl/util/types.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_USE_GNU
// Return x * y as 128-bit integer
// Correctness if x * y < 128 bits
inline uint128_t MultiplyUInt64(uint64_t x, uint64_t y) {
  return uint128_t(x) * uint128_t(y);
}

inline uint64_t BarrettReduce128(uint64_t input_hi, uint64_t input_lo,
                                 uint64_t modulus) {
  HEXL_CHECK(modulus != 0, "modulus == 0")
  uint128_t n = (static_cast<uint128_t>(input_hi) << 64) |
                (static_cast<uint128_t>(input_lo));

  return static_cast<uint64_t>(n % modulus);
  // TODO(fboemer): actually use barrett reduction if performance-critical
}

// Returns low 64bit of 128b/64b where x1=high 64b, x0=low 64b
inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
  uint128_t n =
      (static_cast<uint128_t>(x1) << 64) | (static_cast<uint128_t>(x0));
  uint128_t q = n / y;

  return static_cast<uint64_t>(q);
}

// Multiplies x * y as 128-bit integer.
// @param prod_hi Stores high 64 bits of product
// @param prod_lo Stores low 64 bits of product
inline void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                           uint64_t* prod_lo) {
  uint128_t prod = MultiplyUInt64(x, y);
  *prod_hi = static_cast<uint64_t>(prod >> 64);
  *prod_lo = static_cast<uint64_t>(prod);
}

// Return the high 128 minus BitShift bits of the 128-bit product x * y
template <int BitShift>
inline uint64_t MultiplyUInt64Hi(uint64_t x, uint64_t y) {
  uint128_t product = MultiplyUInt64(x, y);
  return static_cast<uint64_t>(product >> BitShift);
}

// Returns most-significant bit of the input
inline uint64_t MSB(uint64_t input) {
  return static_cast<uint64_t>(std::log2l(input));
}

#define HEXL_LOOP_UNROLL_4 _Pragma("GCC unroll 4")
#define HEXL_LOOP_UNROLL_8 _Pragma("GCC unroll 8")

#endif

}  // namespace hexl
}  // namespace intel
