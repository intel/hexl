// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Performs elementwise modular reduction
/// @param[out] result Stores the result
/// @param[in] operand
/// @param[in] n Number of elements in operand
/// @param[in] modulus Modulus with which to perform modular reduction
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * p) Must be 0, 1, 2 or 4. input_mod_factor=0 means, no
/// knowledge of input range. Barrett reduction will be used in this case.
/// input_mod_factor >= output_mod_factor unless input_mod_factor == 0
/// @param[in] output_mod_factor output elements will be in [0,
/// output_mod_factor * modulus) Must be 1 or 2. For input_mod_factor=0,
/// output_mod_factor will be set to 1.
void EltwiseReduceMod(uint64_t* result, const uint64_t* operand, uint64_t n,
                      uint64_t modulus, uint64_t input_mod_factor,
                      uint64_t output_mod_factor);

}  // namespace hexl
}  // namespace intel
