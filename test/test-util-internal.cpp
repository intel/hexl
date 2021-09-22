// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "hexl/util/aligned-allocator.hpp"
#include "ntt/ntt-internal.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

TEST(GenerateInsecureUniformRandomValue, 10) {
  uint64_t max_value = 10;
  uint64_t min_value = 5;

  bool reached_max = false;
  bool reached_min = false;
  for (size_t i = 0; i < 1000; ++i) {
    uint64_t x = GenerateInsecureUniformRandomValue(min_value, max_value);
    EXPECT_LT(x, max_value);
    EXPECT_GE(x, min_value);
    if (x == min_value) {
      reached_min = true;
    }
    if (x == max_value - 1) {
      reached_max = true;
    }
  }
  EXPECT_TRUE(reached_min);
  EXPECT_TRUE(reached_max);
}

TEST(GenerateInsecureUniformRandomValues, 100) {
  uint64_t max_value = 100;
  uint64_t min_value = 10;
  uint64_t length = 1024;

  AlignedVector64<uint64_t> values =
      GenerateInsecureUniformRandomValues(length, min_value, max_value);
  EXPECT_EQ(values.size(), length);
  EXPECT_TRUE(std::all_of(values.begin(), values.end(), [&](uint64_t x) {
    return (x >= min_value) && (x < max_value);
  }));
  EXPECT_TRUE(std::any_of(values.begin(), values.end(),
                          [&](uint64_t x) { return x = min_value; }));
  EXPECT_TRUE(std::any_of(values.begin(), values.end(),
                          [&](uint64_t x) { return x == max_value - 1; }));
}

}  // namespace hexl
}  // namespace intel
