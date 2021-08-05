// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/number-theory/bit-reverse.hpp"

#include <unordered_set>

#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"
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

  uint64_t log2_size = Log2(size);

  switch (log2_size) {
    case 3: {
      BitReversePairBitwise<3>(input);
      break;
    }
    case 4: {
      BitReversePairBitwise<4>(input);
      break;
    }
    case 5: {
      BitReversePairBitwise<5>(input);
      break;
    }
    case 10: {
      BitReversePairBitwise<10>(input);
      break;
    }
    case 11: {
      BitReversePairBitwise<11>(input);
      break;
    }
    case 12: {
      BitReversePairBitwise<12>(input);
      break;
    }
    case 13: {
      BitReversePairBitwise<13>(input);
      break;
    }
    case 14: {
      BitReversePairBitwise<14>(input);
      break;
    }
    case 15: {
      BitReversePairBitwise<15>(input);
      break;
    }
    case 16: {
      BitReversePairBitwise<16>(input);
      break;
    }
    case 17: {
      BitReversePairBitwise<17>(input);
      break;
    }
    case 18: {
      BitReversePairBitwise<18>(input);
      break;
    }
    default: {
      BitReverseReference(input, size);
    }
  }
}

void BitReverseReference(uint64_t* input, uint64_t size) {
  uint64_t log2_size = Log2(size);
  for (size_t i = 0; i < size; ++i) {
    uint64_t bit_reversed_idx = BitReverseScalar(i, log2_size);
    // Swap only once per pair
    if (i < bit_reversed_idx) {
      std::swap(input[i], input[bit_reversed_idx]);
    }
  }
}

}  // namespace hexl
}  // namespace intel
