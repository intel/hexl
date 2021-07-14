// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/util.hpp"

namespace intel {
namespace hexl {

/// @brief Computes element-wise conditional addition.
/// @param[out] result Stores the result
/// @param[in] operand1 Vector of elements to compare
/// @param[in] n Number of elements in \p operand1
/// @param[in] cmp Comparison operation
/// @param[in] bound Scalar to compare against
/// @param[in] diff Scalar to conditionally add
/// @details Computes result[i] = cmp(operand1[i], bound) ? operand1[i] +
/// diff : operand1[i] for all \f$i=0, ..., n-1\f$.
void EltwiseCmpAddAVX512(uint64_t* result, const uint64_t* operand1, uint64_t n,
                         CMPINT cmp, uint64_t bound, uint64_t diff);

}  // namespace hexl
}  // namespace intel
