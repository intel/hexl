// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"

namespace intel {
namespace hexl {

void BitReverseReference(uint64_t* input, uint64_t size);

inline void BitReverseNative(uint64_t* input, uint64_t size,
                             uint64_t recursion_depth = 0,
                             uint64_t recursion_half = 0) {
  uint64_t bit_width = Log2(size);

  // Base case
  if (bit_width <= recursion_depth) {
    return;
  }

  for (size_t i = 0; i < size; ++i) {
    uint64_t top_swap_idx = bit_width - 1;
    uint64_t bottom_swap_idx = recursion_depth;

    uint64_t bottom_bit = GetBit(i, bottom_swap_idx);
    uint64_t top_bit = GetBit(i, top_swap_idx);

    uint64_t new_idx = i;

    // Clear top and bottom bits
    new_idx = SetBit(new_idx, top_swap_idx, bottom_bit);
    new_idx = SetBit(new_idx, bottom_swap_idx, top_bit);

    if (i < new_idx) {
      std::swap(input[i], input[new_idx]);
    }
  }

  BitReverseNative(input, size / 2, recursion_depth + 1, 2 * recursion_half);
  BitReverseNative(&input[size / 2], size / 2, recursion_depth + 1,
                   2 * recursion_half + 1);
}

}  // namespace hexl
}  // namespace intel
