// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"

namespace intel {
namespace hexl {

/// @brief The Harvey butterfly: assume \p X, \p Y in [0, 4q), and return X', Y'
/// in [0, 4q) such that X' = X + WY, Y' = X - WY (mod q).
/// @param[in,out] X Butterfly data
/// @param[in,out] Y Butterfly data
/// @param[in] W Root of unity
/// @param[in] W_precon Preconditioned \p W for BitShift-bit Barrett
/// reduction
/// @param[in] modulus Negative modulus, i.e. (-q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @details See Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf
inline void FwdButterfly(uint64_t* X, uint64_t* Y, uint64_t W,
                         uint64_t W_precon, uint64_t modulus,
                         uint64_t twice_modulus) {
  HEXL_VLOG(3, "FwdButterfly");
  HEXL_VLOG(3, "Inputs: X " << *X << ", Y " << *Y << ", W " << W << ", modulus "
                            << modulus);
  uint64_t tx = (*X >= twice_modulus) ? (*X - twice_modulus) : *X;
  uint64_t T = MultiplyModLazy<64>(*Y, W, W_precon, modulus);
  HEXL_VLOG(3, "T " << T);
  *X = tx + T;
  *Y = tx + twice_modulus - T;

  HEXL_VLOG(3, "Output X " << *X << ", Y " << *Y);
}

// Assume X, Y in [0, np) and return X', Y' in [0, (n+2)p)
// such that X' = X + WY mod p and Y' = X - WY mod p
inline void FwdButterflyLazy(uint64_t* X, uint64_t* Y, uint64_t W,
                             uint64_t W_precon, uint64_t modulus,
                             uint64_t twice_modulus) {
  HEXL_VLOG(3, "FwdButterflyLazy");
  HEXL_VLOG(3, "Inputs: X " << *X << ", Y " << *Y << ", W " << W << ", modulus "
                            << modulus);

  uint64_t tx = *X;
  uint64_t T = MultiplyModLazy<64>(*Y, W, W_precon, modulus);
  HEXL_VLOG(3, "T " << T);
  *X = tx + T;
  *Y = tx + twice_modulus - T;

  HEXL_VLOG(3, "Outputs: X " << *X << ", Y " << *Y);
}

// Assume X0, X1, X2, X3 in [0, 4q) and return X0, X1, X2, X3 in [0, 4q)
inline void FwdButterflyRadix4(uint64_t* X0, uint64_t* X1, uint64_t* X2,
                               uint64_t* X3, uint64_t W1, uint64_t W2,
                               uint64_t W3, uint64_t modulus,
                               uint64_t twice_modulus) {
  HEXL_VLOG(3, "FwdButterflyRadix4");
  // HEXL_VLOG(3, "Xs " << (std::vector<uint64_t>{*X0, *X1, *X2, *X3}));

  if (std::getenv("TEST") != nullptr) {
    // {
    uint64_t W1_precon = MultiplyFactor(W1, 64, modulus).BarrettFactor();
    uint64_t W2_precon = MultiplyFactor(W2, 64, modulus).BarrettFactor();
    uint64_t W3_precon = MultiplyFactor(W3, 64, modulus).BarrettFactor();

    // Returns Xs in [0, 4p)
    FwdButterflyLazy(X0, X2, W1, W1_precon, modulus, twice_modulus);
    FwdButterflyLazy(X1, X3, W1, W1_precon, modulus, twice_modulus);

    // HEXL_VLOG(3, "W1_precon " << W1_precon);

    // HEXL_VLOG(3, "tmp0 " << *X0);
    // HEXL_VLOG(3, "tmp1 " << *X1);
    // HEXL_VLOG(3, "tmp2 " << *X2);
    // HEXL_VLOG(3, "tmp3 " << *X3);

    // Returns Xs in [0, 8p)
    FwdButterflyLazy(X0, X1, W2, W2_precon, modulus, twice_modulus);
    FwdButterflyLazy(X2, X3, W3, W3_precon, modulus, twice_modulus);

    // HEXL_VLOG(3, "Xs after second round "
    //                  << (std::vector<uint64_t>{*X0, *X1, *X2, *X3}));

    // Reduce Xs to [0, 4p)
    *X0 = ReduceMod<2>(*X0, 4 * modulus);
    *X1 = ReduceMod<2>(*X1, 4 * modulus);
    *X2 = ReduceMod<2>(*X2, 4 * modulus);
    *X3 = ReduceMod<2>(*X3, 4 * modulus);

    return;
  }

  uint64_t a0 = *X0;
  uint64_t a1 = MultiplyMod(*X1, W1, modulus);
  uint64_t a2 = MultiplyMod(*X2, W2, modulus);
  uint64_t a3 = MultiplyMod(*X3, W3, modulus);

  HEXL_VLOG(5, "as " << (std::vector<uint64_t>{a0, a1, a2, a3}));

  uint64_t W1_x_x2 = MultiplyMod(W1, *X2, modulus);
  uint64_t tmp0 = AddUIntMod(*X0, W1_x_x2, modulus);
  uint64_t tmp2 = SubUIntMod(*X0, W1_x_x2, modulus);
  HEXL_VLOG(5, "W1_x_x2 " << W1_x_x2);
  HEXL_VLOG(5, "tmp0 " << tmp0);
  HEXL_VLOG(5, "tmp2 " << tmp2);

  uint64_t W1_x_x3 = MultiplyMod(W1, *X3, modulus);
  HEXL_VLOG(5, "W1_x_x4 " << W1_x_x3);

  uint64_t tmp1 = AddUIntMod(*X1, W1_x_x3, modulus);
  uint64_t tmp3 = SubUIntMod(*X1, W1_x_x3, modulus);
  HEXL_VLOG(5, "tmp1 " << tmp1);
  HEXL_VLOG(5, "tmp3 " << tmp3);

  uint64_t tmp1_x_W2 = MultiplyMod(tmp1, W2, modulus);
  uint64_t Y0 = AddUIntMod(tmp0, tmp1_x_W2, modulus);
  uint64_t Y1 = SubUIntMod(tmp0, tmp1_x_W2, modulus);

  uint64_t tmp3_x_W2 = MultiplyMod(tmp3, W3, modulus);
  uint64_t Y2 = AddUIntMod(tmp2, tmp3_x_W2, modulus);
  uint64_t Y3 = SubUIntMod(tmp2, tmp3_x_W2, modulus);

  *X0 = Y0;
  *X1 = Y1;
  *X2 = Y2;
  *X3 = Y3;

  HEXL_VLOG(5, "Ys " << (std::vector<uint64_t>{Y0, Y1, Y2, Y3}));
}

/// @brief The Harvey butterfly: assume X, Y in [0, 2q), and return X', Y' in
/// [0, 2q) such that X' = X + Y (mod q), Y' = W(X - Y) (mod q).
/// @param[in,out] X Butterfly data
/// @param[in,out] Y Butterfly data
/// @param[in] W Root of unity
/// @param[in] W_precon Preconditioned root of unity for 64-bit Barrett
/// reduction
/// @param[in] neg_modulus Negative modulus, i.e. (-q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @details See Algorithm 3 of https://arxiv.org/pdf/1205.2926.pdf
inline void InvButterfly(uint64_t* X, uint64_t* Y, uint64_t W,
                         uint64_t W_precon, uint64_t modulus,
                         uint64_t twice_modulus) {
  uint64_t tx = *X + *Y;
  uint64_t ty = *X + twice_modulus - *Y;

  *X = (tx >= twice_modulus) ? (tx - twice_modulus) : tx;
  *Y = MultiplyModLazy<64>(ty, W, W_precon, modulus);
}

}  // namespace hexl
}  // namespace intel
