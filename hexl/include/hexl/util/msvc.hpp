// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef HEXL_USE_MSVC

#define NOMINMAX  // Avoid errors with std::min/std::max
#undef min
#undef max

#include <immintrin.h>
#include <intrin.h>
#include <stdint.h>

#include <cmath>

#include "hexl/util/check.hpp"

#pragma intrinsic(_addcarry_u64, _BitScanReverse64, _subborrow_u64, _udiv128, \
                  _umul128)

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

// Multiplies x * y as 128-bit integer.
// @param prod_hi Stores high 64 bits of product
// @param prod_lo Stores low 64 bits of product
inline void MultiplyUInt64(uint64_t x, uint64_t y, uint64_t* prod_hi,
                           uint64_t* prod_lo) {
  *prod_lo = _umul128(x, y, prod_hi);
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

/// @brief Computes Left Shift op as 128-bit unsigned integer
/// @param[out] result_hi Stores high 64 bits of result
/// @param[out] result_lo Stores low 64 bits of result
/// @param[in] op_hi Stores high 64 bits of input
/// @param[in] op_lo Stores low 64 bits of input
inline void LeftShift128(uint64_t* result_hi, uint64_t* result_lo,
                         const uint64_t op_hi, const uint64_t op_lo,
                         const uint64_t shift_value) {
  HEXL_CHECK(result_hi != nullptr, "Require result_hi != nullptr");
  HEXL_CHECK(result_lo != nullptr, "Require result_lo != nullptr");
  HEXL_CHECK(shift_value <= 128,
             "shift_value cannot be greater than 128 " << shift_value);

  if (shift_value == 0) {
    *result_hi = op_hi;
    *result_lo = op_lo;
  } else if (shift_value == 64) {
    *result_hi = op_lo;
    *result_lo = 0ULL;
  } else if (shift_value == 128) {
    *result_hi = 0ULL;
    *result_lo = 0ULL;
  } else if (shift_value >= 1 && shift_value <= 63) {
    *result_hi = (op_hi << shift_value) | (op_lo >> (64 - shift_value));
    *result_lo = op_lo << shift_value;
  } else if (shift_value >= 65 && shift_value < 128) {
    *result_hi = op_lo << (shift_value - 64);
    *result_lo = 0ULL;
  }
}

/// @brief Computes Right Shift op as 128-bit unsigned integer
/// @param[out] result_hi Stores high 64 bits of result
/// @param[out] result_lo Stores low 64 bits of result
/// @param[in] op_hi Stores high 64 bits of input
/// @param[in] op_lo Stores low 64 bits of input
inline void RightShift128(uint64_t* result_hi, uint64_t* result_lo,
                          const uint64_t op_hi, const uint64_t op_lo,
                          const uint64_t shift_value) {
  HEXL_CHECK(result_hi != nullptr, "Require result_hi != nullptr");
  HEXL_CHECK(result_lo != nullptr, "Require result_lo != nullptr");
  HEXL_CHECK(shift_value <= 128,
             "shift_value cannot be greater than 128 " << shift_value);

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

/// @brief Adds op1 + op2 as 128-bit integer
/// @param[out] result_hi Stores high 64 bits of result
/// @param[out] result_lo Stores low 64 bits of result
inline void AddWithCarry128(uint64_t* result_hi, uint64_t* result_lo,
                            const uint64_t op1_hi, const uint64_t op1_lo,
                            const uint64_t op2_hi, const uint64_t op2_lo) {
  HEXL_CHECK(result_hi != nullptr, "Require result_hi != nullptr");
  HEXL_CHECK(result_lo != nullptr, "Require result_lo != nullptr");

  // first 64bit block
  *result_lo = op1_lo + op2_lo;
  unsigned char carry = static_cast<unsigned char>(*result_lo < op1_lo);

  // second 64bit block
  _addcarry_u64(carry, op1_hi, op2_hi, result_hi);
}

/// @brief Subtracts op1 - op2 as 128-bit integer
/// @param[out] result_hi Stores high 64 bits of result
/// @param[out] result_lo Stores low 64 bits of result
inline void SubWithCarry128(uint64_t* result_hi, uint64_t* result_lo,
                            const uint64_t op1_hi, const uint64_t op1_lo,
                            const uint64_t op2_hi, const uint64_t op2_lo) {
  HEXL_CHECK(result_hi != nullptr, "Require result_hi != nullptr");
  HEXL_CHECK(result_lo != nullptr, "Require result_lo != nullptr");

  unsigned char borrow;

  // first 64bit block
  *result_lo = op1_lo - op2_lo;
  borrow = static_cast<unsigned char>(op2_lo > op1_lo);

  // second 64bit block
  _subborrow_u64(borrow, op1_hi, op2_hi, result_hi);
}

/// @brief Computes and returns significant bit count
/// @param[in] value Input element at most 128 bits long
inline uint64_t SignificantBitLength(const uint64_t* value) {
  HEXL_CHECK(value != nullptr, "Require value != nullptr");

  unsigned long count = 0;  // NOLINT(runtime/int)

  // second 64bit block
  _BitScanReverse64(&count, *(value + 1));
  if (count >= 0 && *(value + 1) > 0) {
    return static_cast<uint64_t>(count) + 1 + 64;
  }

  // first 64bit block
  _BitScanReverse64(&count, *value);
  if (count >= 0 && *(value) > 0) {
    return static_cast<uint64_t>(count) + 1;
  }
  return 0;
}

/// @brief Checks if input is negative number
/// @param[in] input Input element to check for sign
inline bool CheckSign(const uint64_t* input) {
  HEXL_CHECK(input != nullptr, "Require input != nullptr");

  uint64_t input_temp[2]{0, 0};
  RightShift128(&input_temp[1], &input_temp[0], input[1], input[0], 127);
  return (input_temp[0] == 1);
}

/// @brief Divides numerator by denominator
/// @param[out] quotient Stores quotient as two 64-bit blocks after division
/// @param[in] numerator
/// @param[in] denominator
inline void DivideUInt128UInt64(uint64_t* quotient, const uint64_t* numerator,
                                const uint64_t denominator) {
  HEXL_CHECK(quotient != nullptr, "Require quotient != nullptr");
  HEXL_CHECK(numerator != nullptr, "Require numerator != nullptr");
  HEXL_CHECK(denominator != 0, "denominator cannot be 0 " << denominator);

  // get bit count of divisor
  uint64_t numerator_bits = SignificantBitLength(numerator);
  const uint64_t numerator_bits_const = numerator_bits;
  const uint64_t uint_128_bit = 128ULL;

  uint64_t MASK[2]{0x0000000000000001, 0x0000000000000000};
  uint64_t remainder[2]{0, 0};
  uint64_t quotient_temp[2]{0, 0};
  uint64_t denominator_temp[2]{denominator, 0};

  quotient[0] = numerator[0];
  quotient[1] = numerator[1];

  // align numerator
  LeftShift128(&quotient[1], &quotient[0], quotient[1], quotient[0],
               (uint_128_bit - numerator_bits_const));

  while (numerator_bits) {
    // if remainder is negative
    if (CheckSign(remainder)) {
      LeftShift128(&remainder[1], &remainder[0], remainder[1], remainder[0], 1);
      RightShift128(&quotient_temp[1], &quotient_temp[0], quotient[1],
                    quotient[0], (uint_128_bit - 1));
      remainder[0] = remainder[0] | quotient_temp[0];
      LeftShift128(&quotient[1], &quotient[0], quotient[1], quotient[0], 1);
      // remainder=remainder+denominator_temp
      AddWithCarry128(&remainder[1], &remainder[0], remainder[1], remainder[0],
                      denominator_temp[1], denominator_temp[0]);
    } else {  // if remainder is positive
      LeftShift128(&remainder[1], &remainder[0], remainder[1], remainder[0], 1);
      RightShift128(&quotient_temp[1], &quotient_temp[0], quotient[1],
                    quotient[0], (uint_128_bit - 1));
      remainder[0] = remainder[0] | quotient_temp[0];
      LeftShift128(&quotient[1], &quotient[0], quotient[1], quotient[0], 1);
      // remainder=remainder-denominator_temp
      SubWithCarry128(&remainder[1], &remainder[0], remainder[1], remainder[0],
                      denominator_temp[1], denominator_temp[0]);
    }

    // if remainder is positive set MSB of quotient[0]=1
    if (!CheckSign(remainder)) {
      MASK[0] = 0x0000000000000001;
      MASK[1] = 0x0000000000000000;
      LeftShift128(&MASK[1], &MASK[0], MASK[1], MASK[0],
                   (uint_128_bit - numerator_bits_const));
      quotient[0] = quotient[0] | MASK[0];
      quotient[1] = quotient[1] | MASK[1];
    }
    quotient_temp[0] = 0;
    quotient_temp[1] = 0;
    numerator_bits--;
  }

  if (CheckSign(remainder)) {
    // remainder=remainder+denominator_temp
    AddWithCarry128(&remainder[1], &remainder[0], remainder[1], remainder[0],
                    denominator_temp[1], denominator_temp[0]);
  }
  RightShift128(&quotient[1], &quotient[0], quotient[1], quotient[0],
                (uint_128_bit - numerator_bits_const));
}

/// @brief Returns low of dividing numerator by denominator
/// @param[in] numerator_hi Stores high 64 bit of numerator
/// @param[in] numerator_lo Stores low 64 bit of numerator
/// @param[in] denominator Stores denominator
inline uint64_t DivideUInt128UInt64Lo(const uint64_t numerator_hi,
                                      const uint64_t numerator_lo,
                                      const uint64_t denominator) {
  uint64_t numerator[2]{numerator_lo, numerator_hi};
  uint64_t quotient[2]{0, 0};

  DivideUInt128UInt64(quotient, numerator, denominator);
  return quotient[0];
}

// Returns most-significant bit of the input
inline uint64_t MSB(uint64_t input) {
  unsigned long index{0};  // NOLINT(runtime/int)
  _BitScanReverse64(&index, input);
  return index;
}

#define HEXL_LOOP_UNROLL_4 \
  {}
#define HEXL_LOOP_UNROLL_8 \
  {}

#endif

}  // namespace hexl
}  // namespace intel
