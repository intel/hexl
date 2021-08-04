// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Returns the input in bit-reversed order
/// @param[in] x Input to reverse
/// @param[in] bit_width Number of bits in the input; must be >= MSB(x)
/// @return The bit-reversed representation of \p x using \p bit_width bits
/// @details For instance, if x = 3, and bit_width = 3, the binary
/// representation of x is 0b011. This function will return the reverse, i.e.
/// 0b110 = 5.
uint64_t BitReverseScalar(uint64_t x, uint64_t bit_width);

void BitReverse(uint64_t* input, uint64_t size);

}  // namespace hexl
}  // namespace intel
