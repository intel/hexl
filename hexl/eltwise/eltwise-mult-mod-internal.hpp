// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"
#include "util/check.hpp"
#include "util/compiler.hpp"

namespace intel {
namespace hexl {

/// @brief Multiplies two vectors elementwise with modular reduction
/// @param[in] result Result of element-wise multiplication
/// @param[in] operand1 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] operand2 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * p) Must be 1, 2 or 4.
/// @details Computes \p result[i] = (\p operand1[i] * \p operand2[i]) mod \p
/// modulus for i=0, ..., \p n - 1
/// @details Algorithm 1 of
/// https://hal.archives-ouvertes.fr/hal-01215845/document
template <int InputModFactor>
void EltwiseMultModNative(uint64_t* result, const uint64_t* operand1,
                          const uint64_t* operand2, uint64_t n,
                          uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 62), "Require modulus < (1ULL << 62)");
  HEXL_CHECK_BOUNDS(operand1, n, InputModFactor * modulus,
                    "operand1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(operand2, n, InputModFactor * modulus,
                    "operand2 exceeds bound " << (InputModFactor * modulus));

  const uint64_t logmod = MSB(modulus);
  // modulus < 2**N
  const uint64_t N = logmod + 1;
  uint64_t L = 63 + N;  // Ensures L - N + 1 == 64
  uint64_t op_hi = uint64_t(1) << (L - 64);
  uint64_t op_lo = uint64_t(0);
  uint64_t barr_lo = DivideUInt128UInt64Lo(op_hi, op_lo, modulus);

  const uint64_t twice_modulus = 2 * modulus;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < n; ++i) {
    uint64_t prod_hi, prod_lo, c2_hi, c2_lo, c4;

    uint64_t x = ReduceMod<InputModFactor>(*operand1, modulus, &twice_modulus);
    uint64_t y = ReduceMod<InputModFactor>(*operand2, modulus, &twice_modulus);

    // Multiply inputs
    MultiplyUInt64(x, y, &prod_hi, &prod_lo);
    // C1 = D >> (N-1)

    uint64_t c1 = (prod_lo >> (N - 1)) + (prod_hi << (64 - (N - 1)));

    // C2 = C1 * barr_lo
    MultiplyUInt64(c1, barr_lo, &c2_hi, &c2_lo);

    // C3 = C2 >> (L - N + 1)
    // L - N + 1 == 64, so we only need high 64 bits
    uint64_t c3 = c2_hi;

    // C4 = prod_lo - (p * c3)_lo
    c4 = prod_lo - c3 * modulus;

    // Conditional subtraction
    *result = (c4 >= modulus) ? (c4 - modulus) : c4;

    ++operand1;
    ++operand2;
    ++result;
  }
}

}  // namespace hexl
}  // namespace intel
