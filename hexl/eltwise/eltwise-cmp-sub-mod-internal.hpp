// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/util.hpp"

namespace intel {
namespace hexl {

/// @brief Computes element-wise conditional modular subtraction.
/// @param[out] result Stores the result
/// @param[in] operand1 Vector of elements to compare
/// @param[in] n Number of elements in \p operand1
/// @param[in] modulus Modulus to reduce by
/// @param[in] cmp Comparison function
/// @param[in] bound Scalar to compare against
/// @param[in] diff Scalar to subtract by
/// @details Computes \p result[i] = (\p cmp(\p operand1, \p bound)) ? (\p
/// operand1 - \p diff) mod \p modulus : \p operand1 for all i=0, ..., n-1
void EltwiseCmpSubModNative(uint64_t* result, const uint64_t* operand1,
                            uint64_t n, uint64_t modulus, CMPINT cmp,
                            uint64_t bound, uint64_t diff);

}  // namespace hexl
}  // namespace intel
