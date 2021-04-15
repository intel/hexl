// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Computes fused multiply-add (\p arg1 * \p arg2 + \p arg3) mod \p
/// modulus element-wise, broadcasting scalars to vectors.
/// @param[out] result Stores the result
/// @param[in] arg1 Vector to multiply
/// @param[in] arg2 Scalar to multiply
/// @param[in] arg3 Vector to add. Will not add if \p arg3 == nullptr
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction. Must be
/// in the range \f$ [2, 2^{61} - 1]\f$
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * modulus). Must be 1, 2, 4, or 8.
void EltwiseFMAMod(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                   const uint64_t* arg3, uint64_t n, uint64_t modulus,
                   uint64_t input_mod_factor);

}  // namespace hexl
}  // namespace intel
