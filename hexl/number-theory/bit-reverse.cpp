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

uint64_t BitReverseScalar(uint64_t x, uint64_t bit_width) {
  HEXL_CHECK(x == 0 || MSB(x) <= bit_width, "MSB(" << x << ") = " << MSB(x)
                                                   << " must be >= bit_width "
                                                   << bit_width)
  if (bit_width == 0) {
    return 0;
  }
  uint64_t rev = 0;
  for (uint64_t i = bit_width; i > 0; i--) {
    rev |= ((x & 1) << (i - 1));
    x >>= 1;
  }
  return rev;
}

void BitReverse(uint64_t* input, uint64_t size) {
  HEXL_CHECK(input != nullptr, "Input cannot be nullptr");
  HEXL_CHECK(IsPowerOfTwo(size), "Size " << size << " must be a power of two");

  BitReverseReference(input, size);
}

void BitReverseReference(uint64_t* input, uint64_t size) {
  uint64_t log2_size = Log2(size);
  for (size_t i = 0; i < size; ++i) {
    uint64_t bit_reversed_idx = BitReverseScalar(i, log2_size);
    if (i < bit_reversed_idx) {
      std::swap(input[i], input[bit_reversed_idx]);
    }
  }
}

}  // namespace hexl
}  // namespace intel
