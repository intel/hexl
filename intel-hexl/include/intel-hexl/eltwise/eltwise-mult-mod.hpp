// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Multiplies two vectors elementwise with modular reduction
/// @param[in] result Result of element-wise multiplication
/// @param[in] operand1 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] operand2 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * p) Must be 1, 2 or 4.
/// @details Computes \p result[i] = (\p operand1[i] * \p operand2[i]) mod \p
/// modulus for i=0, ..., \p n - 1
void EltwiseMultMod(uint64_t* result, const uint64_t* operand1,
                    const uint64_t* operand2, uint64_t n, uint64_t modulus,
                    uint64_t input_mod_factor);

}  // namespace hexl
}  // namespace intel
