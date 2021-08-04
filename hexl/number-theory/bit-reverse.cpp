// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/number-theory/bit-reverse.hpp"

#include <unordered_set>

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "number-theory/bit-reverse-internal.hpp"

namespace intel {
namespace hexl {

void BitReverse(uint64_t* input, uint64_t size) {
  HEXL_CHECK(input != nullptr, "Input cannot be nullptr");
  HEXL_CHECK(IsPowerOfTwo(size), "Size " << size << " must be a power of two");

  BitReverseNative(input, size);
}

void BitReverseNative(uint64_t* input, uint64_t size) {
  std::unordered_set<uint64_t> swapped_indices;

  for (size_t i = 0; i < size; ++i) {
    if (swapped_indices.find(i) == swapped_indices.end()) {
      uint64_t bit_reversed_idx = ReverseBits(i, Log2(size));
      LOG(INFO) << "Swapping " << i << " / " << bit_reversed_idx;
      std::swap(input[i], input[bit_reversed_idx]);
      swapped_indices.insert(bit_reversed_idx);
      swapped_indices.insert(i);
    }
  }
}

}  // namespace hexl
}  // namespace intel
