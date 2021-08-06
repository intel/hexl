// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"

namespace intel {
namespace hexl {

void BitReverseReference(uint64_t* input, uint64_t size);

// Reversing bitwise using pairs of bits; see
// https://arxiv.org/pdf/1708.01873.pdf
//  C++11 doesn't allow function template
// partial specialization, so we wrap this in a class
template <uint64_t BitWidth, uint64_t RecursionDepth>
struct BitReversePairBitwiseHelper {
  static inline void Reverse(uint64_t* input) {
    // uint64_t bit_width = Log2(size);

    // LOG(INFO) << "Reverse BitWidth " << BitWidth;

    // Base case
    if (BitWidth <= RecursionDepth) {
      return;
    }

    constexpr uint64_t top_swap_idx = BitWidth - 1;
    uint64_t bottom_swap_idx = RecursionDepth;

    constexpr uint64_t size_div_two = 1ULL << (BitWidth - 1);

    uint64_t start_idx = 1ULL << RecursionDepth;
    uint64_t block_size = start_idx;
    uint64_t increment = 1ULL << (RecursionDepth + 1);

    // First index doesn't need to be swapped
    for (size_t i = start_idx; i < size_div_two; i += increment) {
      // Loop only through indices with first bit zero and last bit one,
      // where "first" and "last" are given by Recursive depth.
      for (size_t j = 0; j < block_size; ++j) {
        uint64_t index = i + j;

        // XOR temporary
        uint64_t x =
            ((index >> bottom_swap_idx) ^ (index >> top_swap_idx)) & 1ULL;
        uint64_t new_index =
            index ^ ((x << bottom_swap_idx) | (x << top_swap_idx));
        std::swap(input[index], input[new_index]);
      }
    }

    BitReversePairBitwiseHelper<BitWidth - 1, RecursionDepth + 1>::Reverse(
        input);
    BitReversePairBitwiseHelper<BitWidth - 1, RecursionDepth + 1>::Reverse(
        &input[size_div_two]);
  }
};

template <uint64_t RecursionDepth>
struct BitReversePairBitwiseHelper<0, RecursionDepth> {
  static inline void Reverse(uint64_t* input) { return; }
};

template <uint64_t BitWidth>
inline void BitReversePairBitwise(uint64_t* input) {
  // LOG(INFO) << "BitReversePairBitwise< " << BitWidth << ">";
  BitReversePairBitwiseHelper<BitWidth, 0>::Reverse(input);
}

}  // namespace hexl
}  // namespace intel
