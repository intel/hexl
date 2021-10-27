// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/experimental/misc/lr-mat-vec-mult.hpp"

#include <cstring>

#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

// operand1: num_weights x 2 x n x num_moduli
// operand2: num_weights x 2 x n x num_moduli
//
// results:  num_weights x 3 x n x num_moduli
// [num_weights x {x[0].*y[0], x[0].*y[1]+x[1].*y[0], x[1].*y[1]} x num_moduli].
// TODO(@fdiasmor): Ideally, the size of results can be optimized to [3 x n x
// num_moduli].
void LinRegMatrixVectorMultiply(uint64_t* result, const uint64_t* operand1,
                                const uint64_t* operand2, uint64_t n,
                                const uint64_t* moduli, uint64_t num_moduli,
                                uint64_t num_weights) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(moduli != nullptr, "Require moduli != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(num_weights != 0, "Require n != 0");

  // pointer increment to switch to a next polynomial
  size_t poly_size = n * num_moduli;

  // ciphertext increment to switch to the next ciphertext
  size_t cipher_size = 2 * poly_size;

  // ciphertext output increment to switch to the next output
  size_t output_size = 3 * poly_size;

  AlignedVector64<uint64_t> temp(n, 0);

  for (size_t r = 0; r < num_weights; r++) {
    size_t next_output = r * output_size;
    size_t next_poly_pair = r * cipher_size;
    uint64_t* cipher2 = result + next_output;
    const uint64_t* cipher0 = operand1 + next_poly_pair;
    const uint64_t* cipher1 = operand2 + next_poly_pair;

    for (size_t i = 0; i < num_moduli; i++) {
      size_t i_times_n = i * n;
      size_t poly0_offset = i_times_n;
      size_t poly1_offset = poly0_offset + poly_size;
      size_t poly2_offset = poly0_offset + 2 * poly_size;

      // Output ciphertext has 3 polynomials, where x, y are the input
      // ciphertexts: (x[0] * y[0], x[0] * y[1] + x[1] * y[0], x[1] * y[1])

      // Compute third output polynomial
      // Output written directly to result rather than temporary buffer
      // result[2] = x[1] * y[1]
      intel::hexl::EltwiseMultMod(cipher2 + poly2_offset,
                                  cipher0 + poly1_offset,
                                  cipher1 + poly1_offset, n, moduli[i], 1);

      // Compute second output polynomial
      // result[1] = x[1] * y[0]
      intel::hexl::EltwiseMultMod(cipher2 + poly1_offset,
                                  cipher0 + poly1_offset,
                                  cipher1 + poly0_offset, n, moduli[i], 1);

      // result[1] = x[0] * y[1]
      intel::hexl::EltwiseMultMod(temp.data(), cipher0 + poly0_offset,
                                  cipher1 + poly1_offset, n, moduli[i], 1);
      // result[1] += temp_poly
      intel::hexl::EltwiseAddMod(cipher2 + poly1_offset, cipher2 + poly1_offset,
                                 temp.data(), n, moduli[i]);

      // Compute first output polynomial
      // result[0] = x[0] * y[0]
      intel::hexl::EltwiseMultMod(cipher2 + poly0_offset,
                                  cipher0 + poly0_offset,
                                  cipher1 + poly0_offset, n, moduli[i], 1);
    }
  }

  const bool USE_ADDER_TREE = true;
  if (USE_ADDER_TREE) {
    // Accumulate with the adder-tree algorithm in O(logn)
    for (size_t dist = 1; dist < num_weights; dist += dist) {
      size_t step = dist * 2;
      size_t neighbor_cipher_incr = dist * output_size;
      // This loop can leverage parallelism using #pragma unroll
      for (size_t s = 0; s < num_weights; s += step) {
        size_t next_cipher_pair_incr = s * output_size;
        uint64_t* left_cipher = result + next_cipher_pair_incr;
        uint64_t* right_cipher = left_cipher + neighbor_cipher_incr;

        // This loop can leverage parallelism using #pragma unroll
        for (size_t i = 0; i < num_moduli; i++) {
          size_t i_times_n = i * n;
          size_t poly0_offset = i_times_n;
          size_t poly1_offset = poly0_offset + poly_size;
          size_t poly2_offset = poly0_offset + 2 * poly_size;

          // All EltwiseAddMod below can run in parallel
          intel::hexl::EltwiseAddMod(left_cipher + poly0_offset,
                                     right_cipher + poly0_offset,
                                     left_cipher + poly0_offset, n, moduli[i]);
          intel::hexl::EltwiseAddMod(left_cipher + poly1_offset,
                                     right_cipher + poly1_offset,
                                     left_cipher + poly1_offset, n, moduli[i]);
          intel::hexl::EltwiseAddMod(left_cipher + poly2_offset,
                                     right_cipher + poly2_offset,
                                     left_cipher + poly2_offset, n, moduli[i]);
        }
      }
    }
  } else {
    // Accumulate all rows in sequence
    uint64_t* acc = result;
    for (size_t r = 1; r < num_weights; r++) {
      size_t next_cipher = r * output_size;
      acc += next_cipher;
      for (size_t i = 0; i < num_moduli; i++) {
        size_t i_times_n = i * n;
        size_t poly0_offset = i_times_n;
        size_t poly1_offset = poly0_offset + poly_size;
        size_t poly2_offset = poly0_offset + 2 * poly_size;

        // All EltwiseAddMod below can run in parallel

        intel::hexl::EltwiseAddMod(result + poly0_offset, result + poly0_offset,
                                   acc + poly0_offset, n, moduli[i]);
        intel::hexl::EltwiseAddMod(result + poly1_offset, result + poly1_offset,
                                   acc + poly1_offset, n, moduli[i]);
        intel::hexl::EltwiseAddMod(result + poly2_offset, result + poly2_offset,
                                   acc + poly2_offset, n, moduli[i]);
      }
    }
  }
}

}  // namespace hexl
}  // namespace intel
