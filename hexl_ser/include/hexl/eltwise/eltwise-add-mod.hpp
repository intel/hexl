// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Adds two vectors elementwise with modular reduction
/// @param[out] result Stores result
/// @param[in] operand1 Vector of elements to add. Each element must be less
/// than the modulus
/// @param[in] operand2 Vector of elements to add. Each element must be less
/// than the modulus
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction. Must be
/// in the range \f$[2, 2^{63} - 1]\f$
/// @details Computes \f$ operand1[i] = (operand1[i] + operand2[i]) \mod modulus
/// \f$ for \f$ i=0, ..., n-1\f$.
void EltwiseAddMod(uint64_t* result, const uint64_t* operand1,
                   const uint64_t* operand2, uint64_t n, uint64_t modulus);

/// @brief Adds a vector and scalar elementwise with modular reduction
/// @param[out] result Stores result
/// @param[in] operand1 Vector of elements to add. Each element must be less
/// than the modulus
/// @param[in] operand2 Scalar to add. Must be less
/// than the modulus
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction. Must be
/// in the range \f$[2, 2^{63} - 1]\f$
/// @details Computes \f$ operand1[i] = (operand1[i] + operand2) \mod modulus
/// \f$ for \f$ i=0, ..., n-1\f$.
void EltwiseAddMod(uint64_t* result, const uint64_t* operand1,
                   uint64_t operand2, uint64_t n, uint64_t modulus);

}  // namespace hexl
}  // namespace intel
