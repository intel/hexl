// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "ntt/ntt-default.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void ForwardTransformToBitReverseRadix4(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor) {
  HEXL_CHECK(NTT::CheckArguments(n, modulus), "");
  HEXL_CHECK_BOUNDS(operand, n, modulus * input_mod_factor,
                    "operand exceeds bound " << modulus * input_mod_factor);
  HEXL_CHECK(root_of_unity_powers != nullptr,
             "root_of_unity_powers == nullptr");
  HEXL_CHECK(precon_root_of_unity_powers != nullptr,
             "precon_root_of_unity_powers == nullptr");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "input_mod_factor must be 1, 2, or 4; got " << input_mod_factor);
  HEXL_UNUSED(input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
             "output_mod_factor must be 1 or 4; got " << output_mod_factor);

  HEXL_VLOG(3, "modulus " << modulus);
  HEXL_VLOG(3, "n " << n);

  HEXL_VLOG(3, "operand " << std::vector<uint64_t>(operand, operand + n));

  HEXL_VLOG(3, "root_of_unity_powers " << std::vector<uint64_t>(
                   root_of_unity_powers, root_of_unity_powers + n));

  bool is_power_of_4 = IsPowerOfFour(n);

  uint64_t twice_modulus = modulus << 1;
  uint64_t four_times_modulus = modulus << 2;

  // Radix-2 step for non-powers of 4
  if (!is_power_of_4) {
    HEXL_VLOG(3, "Radix 2 step");

    size_t t = (n >> 1);

    const uint64_t W = root_of_unity_powers[1];
    const uint64_t W_precon = precon_root_of_unity_powers[1];

    uint64_t* X_r = result;
    uint64_t* Y_r = X_r + t;
    const uint64_t* X_op = operand;
    const uint64_t* Y_op = X_op + t;

    HEXL_LOOP_UNROLL_8
    for (size_t j = 0; j < t; j++) {
      FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                         twice_modulus);
    }
    // Data in [0, 4q)
    HEXL_VLOG(3, "after radix 2 outputs "
                     << std::vector<uint64_t>(result, result + n));
  }

  uint64_t m_start = 2;
  size_t t = n >> 3;
  if (is_power_of_4) {
    t = n >> 2;

    uint64_t* X_r0 = result;
    uint64_t* X_r1 = X_r0 + t;
    uint64_t* X_r2 = X_r0 + 2 * t;
    uint64_t* X_r3 = X_r0 + 3 * t;
    const uint64_t* X_op0 = operand;
    const uint64_t* X_op1 = operand + t;
    const uint64_t* X_op2 = operand + 2 * t;
    const uint64_t* X_op3 = operand + 3 * t;

    uint64_t W1_ind = 1;
    uint64_t W2_ind = 2 * W1_ind;
    uint64_t W3_ind = 2 * W1_ind + 1;

    const uint64_t W1 = root_of_unity_powers[W1_ind];
    const uint64_t W2 = root_of_unity_powers[W2_ind];
    const uint64_t W3 = root_of_unity_powers[W3_ind];

    const uint64_t W1_precon = precon_root_of_unity_powers[W1_ind];
    const uint64_t W2_precon = precon_root_of_unity_powers[W2_ind];
    const uint64_t W3_precon = precon_root_of_unity_powers[W3_ind];

    switch (t) {
      case 4: {
        FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                           X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                           W3_precon, modulus, twice_modulus,
                           four_times_modulus);
        FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                           X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                           W3_precon, modulus, twice_modulus,
                           four_times_modulus);
        FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                           X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                           W3_precon, modulus, twice_modulus,
                           four_times_modulus);
        FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                           X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                           W3_precon, modulus, twice_modulus,
                           four_times_modulus);
        break;
      }
      case 1: {
        FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                           X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                           W3_precon, modulus, twice_modulus,
                           four_times_modulus);
        break;
      }
      default: {
        for (size_t j = 0; j < t; j += 16) {
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
        }
      }
    }
    t >>= 2;
    m_start = 4;
  }

  // uint64_t m_start = is_power_of_4 ? 1 : 2;
  // size_t t = (n >> m_start) >> 1;

  for (size_t m = m_start; m < n; m <<= 2) {
    HEXL_VLOG(3, "m " << m);

    size_t X0_offset = 0;

    switch (t) {
      case 4: {
        HEXL_LOOP_UNROLL_8
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            X0_offset += 4 * t;
          }
          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = X_r0;
          const uint64_t* X_op1 = X_r1;
          const uint64_t* X_op2 = X_r2;
          const uint64_t* X_op3 = X_r3;

          uint64_t W1_ind = m + i;
          uint64_t W2_ind = 2 * W1_ind;
          uint64_t W3_ind = 2 * W1_ind + 1;

          const uint64_t W1 = root_of_unity_powers[W1_ind];
          const uint64_t W2 = root_of_unity_powers[W2_ind];
          const uint64_t W3 = root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_root_of_unity_powers[W3_ind];

          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
        }
        break;
      }
      case 1: {
        HEXL_LOOP_UNROLL_8
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            X0_offset += 4 * t;
          }
          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = X_r0;
          const uint64_t* X_op1 = X_r1;
          const uint64_t* X_op2 = X_r2;
          const uint64_t* X_op3 = X_r3;

          uint64_t W1_ind = m + i;
          uint64_t W2_ind = 2 * W1_ind;
          uint64_t W3_ind = 2 * W1_ind + 1;

          const uint64_t W1 = root_of_unity_powers[W1_ind];
          const uint64_t W2 = root_of_unity_powers[W2_ind];
          const uint64_t W3 = root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_root_of_unity_powers[W3_ind];

          FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus,
                             four_times_modulus);
        }
        break;
      }
      default: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            X0_offset += 4 * t;
          }
          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = X_r0;
          const uint64_t* X_op1 = X_r1;
          const uint64_t* X_op2 = X_r2;
          const uint64_t* X_op3 = X_r3;

          uint64_t W1_ind = m + i;
          uint64_t W2_ind = 2 * W1_ind;
          uint64_t W3_ind = 2 * W1_ind + 1;

          const uint64_t W1 = root_of_unity_powers[W1_ind];
          const uint64_t W2 = root_of_unity_powers[W2_ind];
          const uint64_t W3 = root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_root_of_unity_powers[W3_ind];

          for (size_t j = 0; j < t; j += 16) {
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
            FwdButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus,
                               four_times_modulus);
          }
        }
      }
    }
    t >>= 2;
  }

  if (output_mod_factor == 1) {
    for (size_t i = 0; i < n; ++i) {
      if (result[i] >= twice_modulus) {
        result[i] -= twice_modulus;
      }
      if (result[i] >= modulus) {
        result[i] -= modulus;
      }
      HEXL_CHECK(result[i] < modulus, "Incorrect modulus reduction in NTT "
                                          << result[i] << " >= " << modulus);
    }
  }

  HEXL_VLOG(3, "outputs " << std::vector<uint64_t>(result, result + n));
}

void InverseTransformFromBitReverseRadix4(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor) {
  HEXL_CHECK(NTT::CheckArguments(n, modulus), "");
  HEXL_CHECK(inv_root_of_unity_powers != nullptr,
             "inv_root_of_unity_powers == nullptr");
  HEXL_CHECK(precon_inv_root_of_unity_powers != nullptr,
             "precon_inv_root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
             "input_mod_factor must be 1 or 2; got " << input_mod_factor);
  HEXL_UNUSED(input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2; got " << output_mod_factor);

  uint64_t twice_modulus = modulus << 1;
  uint64_t n_div_2 = (n >> 1);

  bool is_power_of_4 = IsPowerOfFour(n);
  // Radix-2 step for powers of 4
  if (is_power_of_4) {
    uint64_t* X_r = result;
    uint64_t* Y_r = X_r + 1;
    const uint64_t* X_op = operand;
    const uint64_t* Y_op = X_op + 1;
    const uint64_t* W = inv_root_of_unity_powers + 1;
    const uint64_t* W_precon = precon_inv_root_of_unity_powers + 1;

    HEXL_LOOP_UNROLL_8
    for (size_t j = 0; j < n / 2; j++) {
      InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, *W++, *W_precon++,
                         modulus, twice_modulus);
      X_r++;
      Y_r++;
      X_op++;
      Y_op++;
    }
    // Data in [0, 2q)
  }

  uint64_t m_start = n >> (is_power_of_4 ? 3 : 2);
  size_t t = is_power_of_4 ? 2 : 1;

  size_t w1_root_index = 1 + (is_power_of_4 ? n_div_2 : 0);
  size_t w3_root_index = n_div_2 + 1 + (is_power_of_4 ? (n / 4) : 0);

  HEXL_VLOG(4, "m_start " << m_start);

  for (size_t m = m_start; m > 0; m >>= 2) {
    HEXL_VLOG(4, "m " << m);
    HEXL_VLOG(4, "t " << t);

    size_t X0_offset = 0;

    switch (t) {
      case 1: {
        for (size_t i = 0; i < m; i++) {
          HEXL_VLOG(4, "i " << i);
          if (i != 0) {
            X0_offset += 4 * t;
          }

          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = operand + X0_offset;
          const uint64_t* X_op1 = X_op0 + t;
          const uint64_t* X_op2 = X_op0 + 2 * t;
          const uint64_t* X_op3 = X_op0 + 3 * t;

          uint64_t W1_ind = w1_root_index++;
          uint64_t W2_ind = w1_root_index++;
          uint64_t W3_ind = w3_root_index++;

          const uint64_t W1 = inv_root_of_unity_powers[W1_ind];
          const uint64_t W2 = inv_root_of_unity_powers[W2_ind];
          const uint64_t W3 = inv_root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_inv_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_inv_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_inv_root_of_unity_powers[W3_ind];

          InvButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus);
        }
        break;
      }
      case 4: {
        for (size_t i = 0; i < m; i++) {
          HEXL_VLOG(4, "i " << i);
          if (i != 0) {
            X0_offset += 4 * t;
          }

          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = X_r0;
          const uint64_t* X_op1 = X_r1;
          const uint64_t* X_op2 = X_r2;
          const uint64_t* X_op3 = X_r3;

          uint64_t W1_ind = w1_root_index++;
          uint64_t W2_ind = w1_root_index++;
          uint64_t W3_ind = w3_root_index++;

          const uint64_t W1 = inv_root_of_unity_powers[W1_ind];
          const uint64_t W2 = inv_root_of_unity_powers[W2_ind];
          const uint64_t W3 = inv_root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_inv_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_inv_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_inv_root_of_unity_powers[W3_ind];

          InvButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus);
          InvButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus);
          InvButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus);
          InvButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                             X_op2++, X_op3++, W1, W1_precon, W2, W2_precon, W3,
                             W3_precon, modulus, twice_modulus);
        }
        break;
      }
      default: {
        HEXL_LOOP_UNROLL_4
        for (size_t i = 0; i < m; i++) {
          HEXL_VLOG(4, "i " << i);
          if (i != 0) {
            X0_offset += 4 * t;
          }

          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = X_r0;
          const uint64_t* X_op1 = X_r1;
          const uint64_t* X_op2 = X_r2;
          const uint64_t* X_op3 = X_r3;

          uint64_t W1_ind = w1_root_index++;
          uint64_t W2_ind = w1_root_index++;
          uint64_t W3_ind = w3_root_index++;

          const uint64_t W1 = inv_root_of_unity_powers[W1_ind];
          const uint64_t W2 = inv_root_of_unity_powers[W2_ind];
          const uint64_t W3 = inv_root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_inv_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_inv_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_inv_root_of_unity_powers[W3_ind];

          for (size_t j = 0; j < t; j++) {
            HEXL_VLOG(4, "j " << j);
            InvButterflyRadix4(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++, X_op1++,
                               X_op2++, X_op3++, W1, W1_precon, W2, W2_precon,
                               W3, W3_precon, modulus, twice_modulus);
          }
        }
      }
    }
    t <<= 2;
    w1_root_index += m;
    w3_root_index += m / 2;
  }

  // When M is too short it only needs the final stage butterfly. Copying here
  // in the case of out-of-place.
  if (result != operand && n == 2) {
    std::memcpy(result, operand, n * sizeof(uint64_t));
  }

  HEXL_VLOG(4, "Starting final invNTT stage");
  HEXL_VLOG(4, "operand " << std::vector<uint64_t>(result, result + n));

  // Fold multiplication by N^{-1} to final stage butterfly
  const uint64_t W = inv_root_of_unity_powers[n - 1];
  HEXL_VLOG(4, "final W " << W);

  const uint64_t inv_n = InverseMod(n, modulus);
  uint64_t inv_n_precon = MultiplyFactor(inv_n, 64, modulus).BarrettFactor();
  const uint64_t inv_n_w = MultiplyMod(inv_n, W, modulus);
  uint64_t inv_n_w_precon =
      MultiplyFactor(inv_n_w, 64, modulus).BarrettFactor();

  uint64_t* X = result;
  uint64_t* Y = X + n_div_2;
  for (size_t j = 0; j < n_div_2; ++j) {
    // Assume X, Y in [0, 2q) and compute
    // X' = N^{-1} (X + Y) (mod q)
    // Y' = N^{-1} * W * (X - Y) (mod q)
    // with X', Y' in [0, 2q)
    uint64_t tx = AddUIntMod(X[j], Y[j], twice_modulus);
    uint64_t ty = X[j] + twice_modulus - Y[j];
    X[j] = MultiplyModLazy<64>(tx, inv_n, inv_n_precon, modulus);
    Y[j] = MultiplyModLazy<64>(ty, inv_n_w, inv_n_w_precon, modulus);
  }

  if (output_mod_factor == 1) {
    // Reduce from [0, 2q) to [0,q)
    for (size_t i = 0; i < n; ++i) {
      result[i] = ReduceMod<2>(result[i], modulus);
      HEXL_CHECK(result[i] < modulus, "Incorrect modulus reduction in InvNTT"
                                          << result[i] << " >= " << modulus);
    }
  }
}

}  // namespace hexl
}  // namespace intel
