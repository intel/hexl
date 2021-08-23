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
  HEXL_VLOG(5, "FwdButterfly");
  HEXL_VLOG(5, "Inputs: X " << *X << ", Y " << *Y << ", W " << W << ", modulus "
                            << modulus);
  uint64_t tx = ReduceMod<2>(*X, twice_modulus);
  uint64_t T = MultiplyModLazy<64>(*Y, W, W_precon, modulus);
  HEXL_VLOG(5, "T " << T);
  *X = tx + T;
  *Y = tx + twice_modulus - T;

  HEXL_VLOG(5, "Output X " << *X << ", Y " << *Y);
}

// Assume X, Y in [0, n*q) and return X', Y' in [0, (n+2)*q)
// such that X' = X + WY mod q and Y' = X - WY mod q
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
                               uint64_t* X3, uint64_t W1, uint64_t W1_precon,
                               uint64_t W2, uint64_t W2_precon, uint64_t W3,
                               uint64_t W3_precon, uint64_t modulus,
                               uint64_t twice_modulus,
                               uint64_t four_times_modulus) {
  HEXL_VLOG(3, "FwdButterflyRadix4");
  HEXL_UNUSED(four_times_modulus);

  FwdButterfly(X0, X2, W1, W1_precon, modulus, twice_modulus);
  FwdButterfly(X1, X3, W1, W1_precon, modulus, twice_modulus);
  FwdButterfly(X0, X1, W2, W2_precon, modulus, twice_modulus);
  FwdButterfly(X2, X3, W3, W3_precon, modulus, twice_modulus);

  // Alternate implementation
  // // Returns Xs in [0, 6q)
  // FwdButterflyLazy(X0, X2, W1, W1_precon, modulus, twice_modulus);
  // FwdButterflyLazy(X1, X3, W1, W1_precon, modulus, twice_modulus);

  // // Returns Xs in [0, 8q)
  // FwdButterflyLazy(X0, X1, W2, W2_precon, modulus, twice_modulus);
  // FwdButterflyLazy(X2, X3, W3, W3_precon, modulus, twice_modulus);

  // // Reduce Xs to [0, 4q)
  // *X0 = ReduceMod<2>(*X0, four_times_modulus);
  // *X1 = ReduceMod<2>(*X1, four_times_modulus);
  // *X2 = ReduceMod<2>(*X2, four_times_modulus);
  // *X3 = ReduceMod<2>(*X3, four_times_modulus);
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
  HEXL_VLOG(4, "InvButterfly X " << *X << ", Y " << *Y << " W " << W
                                 << " W_precon " << W_precon << " modulus "
                                 << modulus);
  uint64_t tx = *X + *Y;
  uint64_t ty = *X + twice_modulus - *Y;

  *X = ReduceMod<2>(tx, twice_modulus);
  *Y = MultiplyModLazy<64>(ty, W, W_precon, modulus);

  HEXL_VLOG(4, "InvButterfly returning X " << *X << ", Y " << *Y);
}

// Assume X, Y in [0, n*q) and return X' in [0, 2*n*q), Y' in [0, 2q)
// such that X' = X + Y (mod q), Y' = W(X - Y) (mod q).
inline void InvButterflyLazy(uint64_t* X, uint64_t* Y, uint64_t W,
                             uint64_t W_precon, uint64_t modulus,
                             uint64_t twice_modulus) {
  HEXL_VLOG(3, "InvButterflyLazy");
  HEXL_VLOG(3, "Inputs: X " << *X << ", Y " << *Y << ", W " << W << ", modulus "
                            << modulus);

  uint64_t ty = *X + twice_modulus - *Y;
  *X = *X + *Y;
  // *X = ReduceMod<2>(*X + *Y, twice_modulus);
  *Y = MultiplyModLazy<64>(ty, W, W_precon, modulus);

  HEXL_VLOG(4, "InvButterflyLazy returning X " << *X << ", Y " << *Y);
}

// Assume X0, X1, X2, X3 in [0, 2q) and return X0, X1, X2, X3 in [0, 2q)
inline void InvButterflyRadix4(uint64_t* X0, uint64_t* X1, uint64_t* X2,
                               uint64_t* X3, uint64_t W1, uint64_t W1_precon,
                               uint64_t W2, uint64_t W2_precon, uint64_t W3,
                               uint64_t W3_precon, uint64_t modulus,
                               uint64_t twice_modulus) {
  HEXL_VLOG(4, "InvButterflyRadix4 "  //
                   << "X0 " << *X0 << ", X1 " << *X1 << ", X2 " << *X2 << " X3 "
                   << *X3                                         //
                   << " W1 " << W1 << " W1_precon " << W1_precon  //
                   << " W2 " << W2 << " W2_precon " << W2_precon  //
                   << " W3 " << W3 << " W3_precon " << W3_precon  //
                   << " modulus " << modulus);

  uint64_t Y0 = *X0;
  uint64_t Y1 = *X1;
  uint64_t Y2 = *X2;
  uint64_t Y3 = *X3;

  InvButterfly(&Y0, &Y1, W1, W1_precon, modulus, twice_modulus);
  InvButterfly(&Y2, &Y3, W2, W2_precon, modulus, twice_modulus);
  InvButterfly(&Y0, &Y2, W3, W3_precon, modulus, twice_modulus);
  InvButterfly(&Y1, &Y3, W3, W3_precon, modulus, twice_modulus);

  // Compute X0, X2 in [0, 4q) and X1, X3 in [0, 2q)
  InvButterflyLazy(X0, X1, W1, W1_precon, modulus, twice_modulus);
  InvButterflyLazy(X2, X3, W2, W2_precon, modulus, twice_modulus);
  *X2 = ReduceMod<2>(*X2, modulus);
  // Compute X0 in [0, 8q), X2 in [0, 2q)
  // *X0 = ReduceMod<2>(*X0, modulus);
  InvButterflyLazy(X0, X2, W3, W3_precon, modulus, twice_modulus);
  // Compute X1 in [0, 4q), X3 in [0, 2q)
  InvButterflyLazy(X1, X3, W3, W3_precon, modulus, twice_modulus);

  // Reduce outputs to [0, 2q)
  // Reduce X0 from [0, 8q) to [0, 2q)
  uint64_t four_time_modulus = 2 * twice_modulus;
  *X0 = ReduceMod<4>(*X0, twice_modulus, &four_time_modulus);
  *X1 = ReduceMod<2>(*X1, twice_modulus);
  // *X2 = ReduceMod<2>(*X1, modulus);

  // *X0 = ReduceMod<4>(*X0, twice_modulus, &four_time_modulus);
  // *X1 = ReduceMod<2>(*X1, modulus);
  // *X2 = ReduceMod<2>(*X2, modulus);
  // *X3 = ReduceMod<2>(*X3, modulus);

  HEXL_VLOG(4, "InvButterflyRadix4 returning X0 "
                   << *X0 << ", X1 " << *X1 << ", X2 " << *X2 << " X3 " << *X3);

  HEXL_CHECK(*X0 <= 2 * modulus, "Bad value X0");
  HEXL_CHECK(*X1 <= 2 * modulus, "Bad value X1");
  HEXL_CHECK(*X2 <= 2 * modulus, "Bad value X2");
  HEXL_CHECK(*X3 <= 2 * modulus, "Bad value X3");

  HEXL_CHECK(Y0 == *X0, "X0 ( " << *X0 << " ) != Y0 ( " << Y0 << ")");
  HEXL_CHECK(Y1 == *X1, "X1 ( " << *X1 << " ) != Y1 ( " << Y1 << ")");
  HEXL_CHECK(Y2 == *X2, "X2 ( " << *X2 << " ) != Y2 ( " << Y2 << ")");
  HEXL_CHECK(Y3 == *X3, "X3 ( " << *X3 << " ) != Y3 ( " << Y3 << ")");
}

}  // namespace hexl
}  // namespace intel
