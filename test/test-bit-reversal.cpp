// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "gtest/gtest.h"
#include "hexl/number-theory/bit-reversal.hpp"
#include "hexl/util/compiler.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(BitReversal, bad_input) {
  // Nullptr input
  EXPECT_ANY_THROW(BitReversal(nullptr, 4));

  // Non power-of-two sizes
  std::vector<uint64_t> x{1, 2, 3, 4};
  EXPECT_ANY_THROW(BitReversal(x.data(), 0));
  EXPECT_ANY_THROW(BitReversal(x.data(), 7));
}
#endif

TEST(BitReversal, 4) {
  std::vector<uint64_t> x{1, 2, 3, 4};
  std::vector<uint64_t> expected_out{1, 3, 2, 4};

  BitReversal(x.data(), x.size());

  EXPECT_EQ(x, expected_out);
}

TEST(BitReversal, 8) {
  std::vector<uint64_t> x{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint64_t> expected_out{0, 4, 2, 6, 1, 5, 3, 7};

  BitReversal(x.data(), x.size());

  EXPECT_EQ(x, expected_out);
}

}  // namespace hexl
}  // namespace intel
