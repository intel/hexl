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

void ForwardTransformToBitReverseRadix2(
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

  uint64_t twice_modulus = modulus << 1;
  size_t t = (n >> 1);

  // In case of out-of-place operation do first pass and convert to in-place
  {
    const uint64_t W = root_of_unity_powers[1];
    const uint64_t W_precon = precon_root_of_unity_powers[1];

    uint64_t* X_r = result;
    uint64_t* Y_r = X_r + t;

    const uint64_t* X_op = operand;
    const uint64_t* Y_op = X_op + t;

    // First pass for out-of-order case
    switch (t) {
      case 8: {
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        break;
      }
      case 4: {
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        break;
      }
      case 2: {
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                           twice_modulus);
        break;
      }
      case 1: {
        FwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W, W_precon, modulus,
                           twice_modulus);
        break;
      }
      default: {
        HEXL_LOOP_UNROLL_8
        for (size_t j = 0; j < t; j += 8) {
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
        }
      }
    }
    t >>= 1;
  }

  // Continue with in-place operation
  for (size_t m = 2; m < n; m <<= 1) {
    size_t j1 = 0;
    switch (t) {
      case 8: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = root_of_unity_powers[m + i];
          const uint64_t W_precon = precon_root_of_unity_powers[m + i];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;

          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
        }
        break;
      }
      case 4: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = root_of_unity_powers[m + i];
          const uint64_t W_precon = precon_root_of_unity_powers[m + i];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;

          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
        }
        break;
      }
      case 2: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = root_of_unity_powers[m + i];
          const uint64_t W_precon = precon_root_of_unity_powers[m + i];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;

          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
        }
        break;
      }
      case 1: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = root_of_unity_powers[m + i];
          const uint64_t W_precon = precon_root_of_unity_powers[m + i];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;

          FwdButterflyRadix2(X_r, Y_r, X_op, Y_op, W, W_precon, modulus,
                             twice_modulus);
        }
        break;
      }
      default: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = root_of_unity_powers[m + i];
          const uint64_t W_precon = precon_root_of_unity_powers[m + i];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;

          HEXL_LOOP_UNROLL_8
          for (size_t j = 0; j < t; j += 8) {
            FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            FwdButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
          }
        }
      }
    }
    t >>= 1;
  }
  if (output_mod_factor == 1) {
    for (size_t i = 0; i < n; ++i) {
      result[i] = ReduceMod<4>(result[i], modulus, &twice_modulus);
      HEXL_CHECK(result[i] < modulus, "Incorrect modulus reduction in NTT "
                                          << result[i] << " >= " << modulus);
    }
  }
}

void ReferenceForwardTransformToBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers) {
  HEXL_CHECK(NTT::CheckArguments(n, modulus), "");
  HEXL_CHECK(root_of_unity_powers != nullptr,
             "root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");

  size_t t = (n >> 1);
  for (size_t m = 1; m < n; m <<= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++) {
      size_t j2 = j1 + t;
      const uint64_t W = root_of_unity_powers[m + i];

      uint64_t* X = operand + j1;
      uint64_t* Y = X + t;
      for (size_t j = j1; j < j2; j++) {
        // X', Y' = X + WY, X - WY (mod q).
        uint64_t tx = *X;
        uint64_t W_x_Y = MultiplyMod(*Y, W, modulus);
        *X++ = AddUIntMod(tx, W_x_Y, modulus);
        *Y++ = SubUIntMod(tx, W_x_Y, modulus);
      }
      j1 += (t << 1);
    }
    t >>= 1;
  }
}

void ReferenceInverseTransformFromBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers) {
  HEXL_CHECK(NTT::CheckArguments(n, modulus), "");
  HEXL_CHECK(inv_root_of_unity_powers != nullptr,
             "inv_root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");

  size_t t = 1;
  size_t root_index = 1;
  for (size_t m = (n >> 1); m >= 1; m >>= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++, root_index++) {
      const uint64_t W = inv_root_of_unity_powers[root_index];
      uint64_t* X_r = operand + j1;
      uint64_t* Y_r = X_r + t;
      for (size_t j = 0; j < t; j++) {
        uint64_t X_op = *X_r;
        uint64_t Y_op = *Y_r;
        // Butterfly X' = (X + Y) mod q, Y' = W(X-Y) mod q
        *X_r = AddUIntMod(X_op, Y_op, modulus);
        *Y_r = MultiplyMod(W, SubUIntMod(X_op, Y_op, modulus), modulus);
        X_r++;
        Y_r++;
      }
      j1 += (t << 1);
    }
    t <<= 1;
  }

  // Final multiplication by N^{-1}
  const uint64_t inv_n = InverseMod(n, modulus);
  for (size_t i = 0; i < n; ++i) {
    operand[i] = MultiplyMod(operand[i], inv_n, modulus);
  }
}

void InverseTransformFromBitReverseRadix2(
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
  size_t t = 1;
  size_t root_index = 1;

  for (size_t m = n_div_2; m > 1; m >>= 1) {
    size_t j1 = 0;

    switch (t) {
      case 1: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = inv_root_of_unity_powers[root_index];
          const uint64_t W_precon = precon_inv_root_of_unity_powers[root_index];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = operand + j1;
          const uint64_t* Y_op = X_op + t;
          InvButterflyRadix2(X_r, Y_r, X_op, Y_op, W, W_precon, modulus,
                             twice_modulus);
        }
        break;
      }
      case 2: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = inv_root_of_unity_powers[root_index];
          const uint64_t W_precon = precon_inv_root_of_unity_powers[root_index];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
        }
        break;
      }
      case 4: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = inv_root_of_unity_powers[root_index];
          const uint64_t W_precon = precon_inv_root_of_unity_powers[root_index];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
        }
        break;
      }
      case 8: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = inv_root_of_unity_powers[root_index];
          const uint64_t W_precon = precon_inv_root_of_unity_powers[root_index];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
          InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon, modulus,
                             twice_modulus);
        }
        break;
      }
      default: {
        for (size_t i = 0; i < m; i++, root_index++) {
          if (i != 0) {
            j1 += (t << 1);
          }
          const uint64_t W = inv_root_of_unity_powers[root_index];
          const uint64_t W_precon = precon_inv_root_of_unity_powers[root_index];

          uint64_t* X_r = result + j1;
          uint64_t* Y_r = X_r + t;
          const uint64_t* X_op = X_r;
          const uint64_t* Y_op = Y_r;

          HEXL_LOOP_UNROLL_8
          for (size_t j = 0; j < t; j += 8) {
            InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
            InvButterflyRadix2(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
                               modulus, twice_modulus);
          }
        }
      }
    }
    t <<= 1;
  }

  // When M is too short it only needs the final stage butterfly. Copying here
  // in the case of out-of-place.
  if (result != operand && n == 2) {
    std::memcpy(result, operand, n * sizeof(uint64_t));
  }

  // Fold multiplication by N^{-1} to final stage butterfly
  const uint64_t W = inv_root_of_unity_powers[n - 1];

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
