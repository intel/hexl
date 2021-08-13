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
  uint64_t tx = (*X >= twice_modulus) ? (*X - twice_modulus) : *X;
  uint64_t T = MultiplyModLazy<64>(*Y, W, W_precon, modulus);
  *X = tx + T;
  *Y = tx + twice_modulus - T;
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
