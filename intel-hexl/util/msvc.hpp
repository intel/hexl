// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef HEXL_USE_MSVC
#include <immintrin.h>
#include <intrin.h>
#include <stdint.h>

#include <iostream>

#pragma intrinsic(_udiv128, _umul128)

#undef TRUE
#undef FALSE

namespace intel {
namespace hexl {

inline uint64_t BarrettReduce128(uint64_t input_hi, uint64_t input_lo,
                                 uint64_t modulus) {
  HEXL_CHECK(modulus != 0, "modulus == 0")
  uint64_t remainder;
  _udiv128(input_hi, input_lo, modulus, &remainder);

  return remainder;
}

// Returns low 64bit of 128b/64b where x1=high 64b, x0=low 64b
inline uint64_t DivideUInt128UInt64Lo(uint64_t x1, uint64_t x0, uint64_t y) {
  uint64_t remainder;
  uint64_t result = _udiv128(x1, x0, y, &remainder);
  return result;
}

// Multiplies x * y as 128-bit integer.
// @param prod_hi Stores high 64 bits of product
// @param prod_lo Stores low 64 bits of product
inline void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                           uint64_t* prod_lo) {
  *prod_lo = _umul128(x, y, prod_hi);
}

inline void RightShift128(uint64_t* result_hi, uint64_t* result_lo,
                          uint64_t op_hi, uint64_t op_lo,
                          uint64_t shift_value) {
  if (shift_value == 0) {
    *result_hi = op_hi;
    *result_lo = op_lo;
  } else if (shift_value == 64) {
    *result_hi = 0ULL;
    *result_lo = op_hi;
  } else if (shift_value == 128) {
    *result_hi = 0ULL;
    *result_lo = 0ULL;
  } else if (shift_value >= 1 && shift_value <= 63) {
    *result_hi = op_hi >> shift_value;
    *result_lo = (op_hi << (64 - shift_value)) | (op_lo >> shift_value);
  } else if (shift_value >= 65 && shift_value < 128) {
    *result_hi = 0ULL;
    *result_lo = op_hi >> (shift_value - 64);
  }
}

// Return the high 128 minus BitShift bits of the 128-bit product x * y
template <int BitShift>
inline uint64_t MultiplyUInt64Hi(uint64_t x, uint64_t y) {
  HEXL_CHECK(BitShift == 52 || BitShift == 64,
             "Invalid BitShift " << BitShift << "; expected 52 or 64");
  uint64_t prod_hi;
  uint64_t prod_lo = _umul128(x, y, &prod_hi);
  uint64_t result_hi;
  uint64_t result_lo;
  RightShift128(&result_hi, &result_lo, prod_hi, prod_lo, BitShift);
  return result_lo;
}

#define HEXL_LOOP_UNROLL_4 \
  {}
#define HEXL_LOOP_UNROLL_8 \
  {}

#endif

}  // namespace hexl
}  // namespace intel
