// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"

namespace intel {
namespace hexl {

void BitReverseReference(uint64_t* input, uint64_t size);

// Reversing bitwise using pairs of bits
inline void BitReverseNative(uint64_t* input, uint64_t size, uint64_t bit_width,
                             uint64_t recursion_depth = 0) {
  // uint64_t bit_width = Log2(size);

  // Base case
  if (bit_width <= recursion_depth) {
    return;
  }

  uint64_t top_swap_idx = bit_width - 1;
  uint64_t bottom_swap_idx = recursion_depth;

  for (size_t i = 0; i < size; ++i) {
    // XOR temporary
    uint64_t x =
        ((i >> bottom_swap_idx) ^ (i >> top_swap_idx)) & ((1ULL << 1) - 1);
    uint64_t r = i ^ ((x << bottom_swap_idx) | (x << top_swap_idx));

    if (i < r) {
      std::swap(input[i], input[r]);
    }
  }

  BitReverseNative(input, size / 2, bit_width - 1, recursion_depth + 1);
  BitReverseNative(&input[size / 2], size / 2, bit_width - 1,
                   recursion_depth + 1);
}

}  // namespace hexl
}  // namespace intel
