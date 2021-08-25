// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/experimental/seal/ckks-multiply.hpp"

#include <cstring>

#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void CkksMultiply(uint64_t* result, const uint64_t* operand1,
                  const uint64_t* operand2, uint64_t n, const uint64_t* moduli,
                  uint64_t num_moduli) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(moduli != nullptr, "Require moduli != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");

  // pointer increment to switch to a next polynomial
  size_t poly_size = n * num_moduli;

  // Output ciphertext has 3 polynomials, where x, y are the input
  // ciphertexts: (x[0] * y[0], x[0] * y[1] + x[1] * y[0], x[1] * y[1])

  // TODO(fboemer): Determine based on cpu cache size
  size_t tile_size = std::min(n, uint64_t(512));
  size_t num_tiles = n / tile_size;

  AlignedVector64<uint64_t> temp(tile_size, 0);

  // Modulus by modulus
  for (size_t i = 0; i < num_moduli; i++) {
    // Split by tiles for better caching
    size_t i_times_n = i * n;
    for (size_t tile = 0; tile < num_tiles; ++tile) {
      size_t poly0_offset = i_times_n + tile_size * tile;
      size_t poly1_offset = poly0_offset + poly_size;
      size_t poly2_offset = poly0_offset + 2 * poly_size;

      // Compute third output polynomial
      // Output written directly to result rather than temporary buffer
      // result[2] = x[1] * y[1]
      intel::hexl::EltwiseMultMod(
          &result[poly2_offset], operand1 + poly1_offset,
          operand2 + poly1_offset, tile_size, moduli[i], 1);

      // Compute second output polynomial
      // result[1] = x[1] * y[0]
      intel::hexl::EltwiseMultMod(
          &result[poly1_offset], operand1 + poly1_offset,
          operand2 + poly0_offset, tile_size, moduli[i], 1);
      // result[1] = x[0] * y[1]
      intel::hexl::EltwiseMultMod(temp.data(), operand1 + poly0_offset,
                                  operand2 + poly1_offset, tile_size, moduli[i],
                                  1);
      // result[1] += temp_poly
      intel::hexl::EltwiseAddMod(&result[poly1_offset], &result[poly1_offset],
                                 temp.data(), tile_size, moduli[i]);

      // Compute first output polynomial
      // result[0] = x[0] * y[0]
      intel::hexl::EltwiseMultMod(
          &result[poly0_offset], operand1 + poly0_offset,
          operand2 + poly0_offset, tile_size, moduli[i], 1);
    }
  }
}

}  // namespace hexl
}  // namespace intel
