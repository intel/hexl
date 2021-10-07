// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"

namespace intel {
namespace hexl {

/// @brief Out of place Harvey butterfly: assume \p X_op, \p Y_op in [0, 4q),
/// and return X_r, Y_r in [0, 4q) such that X_r = X_op + WY_op, Y_r = X_op -
/// WY_op (mod q).
/// @param[out] X_r Butterfly data
/// @param[out] Y_r Butterfly data
/// @param[in] X_op Butterfly data
/// @param[in] Y_op Butterfly data
/// @param[in] W Root of unity
/// @param[in] W_precon Preconditioned \p W for BitShift-bit Barrett
/// reduction
/// @param[in] modulus Negative modulus, i.e. (-q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @details See Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf
inline void FwdButterflyRadix2(uint64_t* X_r, uint64_t* Y_r,
                               const uint64_t* X_op, const uint64_t* Y_op,
                               uint64_t W, uint64_t W_precon, uint64_t modulus,
                               uint64_t twice_modulus) {
  HEXL_VLOG(5, "FwdButterflyRadix2");
  HEXL_VLOG(5, "Inputs: X_op " << *X_op << ", Y_op " << *Y_op << ", W " << W
                               << ", modulus " << modulus);
  uint64_t tx = ReduceMod<2>(*X_op, twice_modulus);
  uint64_t T = MultiplyModLazy<64>(*Y_op, W, W_precon, modulus);
  HEXL_VLOG(5, "T " << T);
  *X_r = tx + T;
  *Y_r = tx + twice_modulus - T;

  HEXL_VLOG(5, "Output X " << *X_r << ", Y " << *Y_r);
}

// Assume X, Y in [0, n*q) and return X_r, Y_r in [0, (n+2)*q)
// such that X_r = X_op + WY_op mod q and Y_r = X_op - WY_op mod q
inline void FwdButterflyRadix4Lazy(uint64_t* X_r, uint64_t* Y_r,
                                   const uint64_t X_op, const uint64_t Y_op,
                                   uint64_t W, uint64_t W_precon,
                                   uint64_t modulus, uint64_t twice_modulus) {
  HEXL_VLOG(3, "FwdButterflyRadix4Lazy");
  HEXL_VLOG(3, "Inputs: X_op " << X_op << ", Y_op " << Y_op << ", W " << W
                               << ", modulus " << modulus);

  uint64_t T = MultiplyModLazy<64>(Y_op, W, W_precon, modulus);
  HEXL_VLOG(3, "T " << T);
  *X_r = X_op + T;
  *Y_r = X_op + twice_modulus - T;

  HEXL_VLOG(3, "Outputs: X_r " << *X_r << ", Y_r " << *Y_r);
}

// Assume X0, X1, X2, X3 in [0, 4q) and return X0, X1, X2, X3 in [0, 4q)
inline void FwdButterflyRadix4(
    uint64_t* X_r0, uint64_t* X_r1, uint64_t* X_r2, uint64_t* X_r3,
    const uint64_t* X_op0, const uint64_t* X_op1, const uint64_t* X_op2,
    const uint64_t* X_op3, uint64_t W1, uint64_t W1_precon, uint64_t W2,
    uint64_t W2_precon, uint64_t W3, uint64_t W3_precon, uint64_t modulus,
    uint64_t twice_modulus, uint64_t four_times_modulus) {
  HEXL_VLOG(3, "FwdButterflyRadix4");
  HEXL_UNUSED(four_times_modulus);

  FwdButterflyRadix2(X_r0, X_r2, X_op0, X_op2, W1, W1_precon, modulus,
                     twice_modulus);
  FwdButterflyRadix2(X_r1, X_r3, X_op1, X_op3, W1, W1_precon, modulus,
                     twice_modulus);
  FwdButterflyRadix2(X_r0, X_r1, X_r0, X_r1, W2, W2_precon, modulus,
                     twice_modulus);
  FwdButterflyRadix2(X_r2, X_r3, X_r2, X_r3, W3, W3_precon, modulus,
                     twice_modulus);

  // Alternate implementation
  // // Returns Xs in [0, 6q)
  // FwdButterflyRadix4Lazy(X0, X2, W1, W1_precon, modulus, twice_modulus);
  // FwdButterflyRadix4Lazy(X1, X3, W1, W1_precon, modulus, twice_modulus);

  // // Returns Xs in [0, 8q)
  // FwdButterflyRadix4Lazy(X0, X1, W2, W2_precon, modulus, twice_modulus);
  // FwdButterflyRadix4Lazy(X2, X3, W3, W3_precon, modulus, twice_modulus);

  // // Reduce Xs to [0, 4q)
  // *X0 = ReduceMod<2>(*X0, four_times_modulus);
  // *X1 = ReduceMod<2>(*X1, four_times_modulus);
  // *X2 = ReduceMod<2>(*X2, four_times_modulus);
  // *X3 = ReduceMod<2>(*X3, four_times_modulus);
}

/// @brief Out-of-place Harvey butterfly: assume X_op, Y_op in [0, 2q), and
/// return X_r, Y_r in [0, 2q) such that X_r = X_op + Y_op (mod q),
/// Y_r = W(X_op - Y_op) (mod q).
/// @param[out] X_r Butterfly data
/// @param[out] Y_r Butterfly data
/// @param[in] X_op Butterfly data
/// @param[in] Y_op Butterfly data
/// @param[in] W Root of unity
/// @param[in] W_precon Preconditioned root of unity for 64-bit Barrett
/// reduction
/// @param[in] modulus Modulus, i.e. (q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @details See Algorithm 3 of https://arxiv.org/pdf/1205.2926.pdf
inline void InvButterflyRadix2(uint64_t* X_r, uint64_t* Y_r,
                               const uint64_t* X_op, const uint64_t* Y_op,
                               uint64_t W, uint64_t W_precon, uint64_t modulus,
                               uint64_t twice_modulus) {
  HEXL_VLOG(4, "InvButterflyRadix2 X_op "
                   << *X_op << ", Y_op " << *Y_op << " W " << W << " W_precon "
                   << W_precon << " modulus " << modulus);
  uint64_t tx = *X_op + *Y_op;
  *Y_r = *X_op + twice_modulus - *Y_op;
  *X_r = ReduceMod<2>(tx, twice_modulus);
  *Y_r = MultiplyModLazy<64>(*Y_r, W, W_precon, modulus);

  HEXL_VLOG(4, "InvButterflyRadix2 returning X_r " << *X_r << ", Y_r " << *Y_r);
}

// Assume X0, X1, X2, X3 in [0, 2q) and return X0, X1, X2, X3 in [0, 2q)
inline void InvButterflyRadix4(uint64_t* X_r0, uint64_t* X_r1, uint64_t* X_r2,
                               uint64_t* X_r3, const uint64_t* X_op0,
                               const uint64_t* X_op1, const uint64_t* X_op2,
                               const uint64_t* X_op3, uint64_t W1,
                               uint64_t W1_precon, uint64_t W2,
                               uint64_t W2_precon, uint64_t W3,
                               uint64_t W3_precon, uint64_t modulus,
                               uint64_t twice_modulus) {
  HEXL_VLOG(4, "InvButterflyRadix4 "                               //
                   << "X_op0 " << *X_op0 << ", X_op1 " << *X_op1   //
                   << ", X_op2 " << *X_op2 << " X_op3 " << *X_op3  //
                   << " W1 " << W1 << " W1_precon " << W1_precon   //
                   << " W2 " << W2 << " W2_precon " << W2_precon   //
                   << " W3 " << W3 << " W3_precon " << W3_precon   //
                   << " modulus " << modulus);

  InvButterflyRadix2(X_r0, X_r1, X_op0, X_op1, W1, W1_precon, modulus,
                     twice_modulus);
  InvButterflyRadix2(X_r2, X_r3, X_op2, X_op3, W2, W2_precon, modulus,
                     twice_modulus);
  InvButterflyRadix2(X_r0, X_r2, X_r0, X_r2, W3, W3_precon, modulus,
                     twice_modulus);
  InvButterflyRadix2(X_r1, X_r3, X_r1, X_r3, W3, W3_precon, modulus,
                     twice_modulus);

  HEXL_VLOG(4, "InvButterflyRadix4 returning X0 " << *X_r0 << ", X_r1 " << *X_r1
                                                  << ", X_r2 " << *X_r2  //
                                                  << " X_r3 " << *X_r3);
}

}  // namespace hexl
}  // namespace intel
