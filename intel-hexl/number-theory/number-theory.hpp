// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <iostream>
#include <limits>
#include <vector>

#include "util/check.hpp"
#include "util/compiler.hpp"

namespace intel {
namespace hexl {

// Stores an integer on which modular multiplication can be performed more
// efficiently, at the cost of some precomputation.
class MultiplyFactor {
 public:
  MultiplyFactor() = default;

  // Computes and stores the Barrett factor (operand << bit_shift) / modulus
  MultiplyFactor(uint64_t operand, uint64_t bit_shift, uint64_t modulus)
      : m_operand(operand) {
    HEXL_CHECK(operand <= modulus, "operand " << operand
                                              << " must be less than modulus "
                                              << modulus);
    HEXL_CHECK(bit_shift == 64 || bit_shift == 52,
               "Unsupport BitShift " << bit_shift);
    uint64_t op_hi{0};
    uint64_t op_lo{0};

    if (bit_shift == 64) {
      op_hi = operand;
      op_lo = 0;
    } else if (bit_shift == 52) {
      op_hi = operand >> 12;
      op_lo = operand << 52;
    }
    m_barrett_factor = DivideUInt128UInt64Lo(op_hi, op_lo, modulus);
  }

  inline uint64_t BarrettFactor() const { return m_barrett_factor; }
  inline uint64_t Operand() const { return m_operand; }

 private:
  uint64_t m_operand;
  uint64_t m_barrett_factor;
};

// Returns whether or not num is a power of two
inline bool IsPowerOfTwo(uint64_t num) { return num && !(num & (num - 1)); }

// Returns log2(x) for x a power of 2
inline uint64_t Log2(uint64_t x) {
  HEXL_CHECK(IsPowerOfTwo(x), x << " not a power of 2");
  uint64_t ret = 0;
  while (x >>= 1) ++ret;
  return ret;
}

// Returns the maximum value that can be represented using bits bits
inline uint64_t MaximumValue(uint64_t bits) {
  HEXL_CHECK(bits <= 64, "MaximumValue requires bits <= 64; got " << bits);
  if (bits == 64) {
    return (std::numeric_limits<uint64_t>::max)();
  }
  return (1ULL << bits) - 1;
}

// Reverses the bits
uint64_t ReverseBitsUInt(uint64_t x, uint64_t bits);

// Returns a^{-1} mod modulus
uint64_t InverseUIntMod(uint64_t a, uint64_t modulus);

//// Returns (x * y) mod modulus
//// Assumes x, y < modulus
uint64_t MultiplyUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

// Returns (x * y) mod modulus
// @param y_precon floor(2**64 / modulus)
uint64_t MultiplyMod(uint64_t x, uint64_t y, uint64_t y_precon,
                     uint64_t modulus);

// Returns (x + y) mod modulus
// Assumes x, y < modulus
uint64_t AddUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

// Returns (x - y) mod modulus
// Assumes x, y < modulus
uint64_t SubUIntMod(uint64_t x, uint64_t y, uint64_t modulus);

// Returns base^exp mod modulus
uint64_t PowMod(uint64_t base, uint64_t exp, uint64_t modulus);

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
bool IsPrimitiveRoot(uint64_t root, uint64_t degree, uint64_t modulus);

// Tries to return a primtiive degree-th root of unity
// Returns -1 if no root is found
uint64_t GeneratePrimitiveRoot(uint64_t degree, uint64_t modulus);

// Returns true whether root is a degree-th root of unity
// degree must be a power of two.
uint64_t MinimalPrimitiveRoot(uint64_t degree, uint64_t modulus);

// Computes (x * y) mod modulus, except that the output is in [0, 2 * modulus]
// @param modulus_precon Pre-computed Barrett reduction factor
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y_operand,
                                    uint64_t y_barrett_factor,
                                    uint64_t modulus) {
  HEXL_CHECK(y_operand < modulus, "y_operand " << y_operand
                                               << " must be less than modulus "
                                               << modulus);
  HEXL_CHECK(
      modulus <= MaximumValue(BitShift),
      "Modulus " << modulus << " exceeds bound " << MaximumValue(BitShift));
  HEXL_CHECK(x <= MaximumValue(BitShift),
             "Operand " << x << " exceeds bound " << MaximumValue(BitShift));

  uint64_t Q = MultiplyUInt64Hi<BitShift>(x, y_barrett_factor);
  return y_operand * x - Q * modulus;
}

// Computes (x * y) mod modulus, except that the output is in [0, 2 * modulus]
template <int BitShift>
inline uint64_t MultiplyUIntModLazy(uint64_t x, uint64_t y, uint64_t modulus) {
  HEXL_CHECK(BitShift == 64 || BitShift == 52,
             "Unsupport BitShift " << BitShift);
  HEXL_CHECK(x <= MaximumValue(BitShift),
             "Operand " << x << " exceeds bound " << MaximumValue(BitShift));
  HEXL_CHECK(y < modulus,
             "y " << y << " must be less than modulus " << modulus);
  HEXL_CHECK(
      modulus <= MaximumValue(BitShift),
      "Modulus " << modulus << " exceeds bound " << MaximumValue(BitShift));
  uint64_t y_hi{0};
  uint64_t y_lo{0};
  if (BitShift == 64) {
    y_hi = y;
    y_lo = 0;
  } else if (BitShift == 52) {
    y_hi = y >> 12;
    y_lo = y << 52;
  }
  uint64_t y_barrett = DivideUInt128UInt64Lo(y_hi, y_lo, modulus);
  return MultiplyUIntModLazy<BitShift>(x, y, y_barrett, modulus);
}

// Adds two unsigned 64-bit integers
// @param operand1 Number to add
// @param operand2 Number to add
// @param result Stores the sum
// @return The carry bit
inline unsigned char AddUInt64(uint64_t operand1, uint64_t operand2,
                               uint64_t* result) {
  *result = operand1 + operand2;
  return static_cast<unsigned char>(*result < operand1);
}

// Returns whether or not the input is prime
bool IsPrime(uint64_t n);

// Generates a list of num_primes primes in the range [2^(bit_size,
// 2^(bit_size+1)]. Ensures each prime p satisfies
// p % (2*ntt_size+1)) == 1
// @param num_primes Number of primes to generate
// @param bit_size Bit size of each prime
// @param ntt_size N such that each prime p satisfies p % (2N) == 1. N must be
// a power of two
std::vector<uint64_t> GeneratePrimes(size_t num_primes, size_t bit_size,
                                     size_t ntt_size = 1);

// returns input mod modulus, computed via Barrett reduction
// @param p_barr floor(2^64 / p)
uint64_t BarrettReduce64(uint64_t input, uint64_t modulus, uint64_t p_barr);

template <int InputModFactor>
uint64_t ReduceMod(uint64_t x, uint64_t modulus,
                   const uint64_t* twice_modulus = nullptr,
                   const uint64_t* four_times_modulus = nullptr) {
  HEXL_CHECK(InputModFactor == 1 || InputModFactor == 2 ||
                 InputModFactor == 4 || InputModFactor == 8,
             "InputModFactor should be 1, 2, 4, or 8");
  if (InputModFactor == 1) {
    return x;
  }
  if (InputModFactor == 2) {
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  if (InputModFactor == 4) {
    HEXL_CHECK(twice_modulus != nullptr, "twice_modulus should not be nullptr");
    if (x >= *twice_modulus) {
      x -= *twice_modulus;
    }
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  if (InputModFactor == 8) {
    HEXL_CHECK(twice_modulus != nullptr, "twice_modulus should not be nullptr");
    HEXL_CHECK(four_times_modulus != nullptr,
               "four_times_modulus should not be nullptr");

    if (x >= *four_times_modulus) {
      x -= *four_times_modulus;
    }
    if (x >= *twice_modulus) {
      x -= *twice_modulus;
    }
    if (x >= modulus) {
      x -= modulus;
    }
    return x;
  }
  HEXL_CHECK(false, "Should be unreachable");
  return x;
}

}  // namespace hexl
}  // namespace intel
