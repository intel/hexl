// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Adds two vectors elementwise with modular reduction
/// @param[out] result Stores result
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction. Must be
/// in the range \f$[2, 2^{63} - 1]\f$
/// @details Computes \f$ result[i] = (operand1[i] * operand2[i]) + (operand3[i]
/// * operand4[i]) \mod modulus \f$ for \f$ i=0, ..., n-1\f$.
void EltwiseDotMod(uint64_t* result, const uint64_t* operand1,
                   const uint64_t* operand2, const uint64_t* operand3,
                   const uint64_t* operand4, uint64_t n, uint64_t modulus);

}  // namespace hexl
}  // namespace intel
