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
  uint64_t modulus = 10;

  bool reached_max = false;
  for (size_t i = 0; i < 1000; ++i) {
    uint64_t x = GenerateInsecureUniformRandomValue(modulus);
    EXPECT_LT(x, modulus);
    if (x == modulus - 1) {
      reached_max = true;
    }
  }
  EXPECT_TRUE(reached_max);
}

TEST(GenerateInsecureUniformRandomValues, 100) {
  uint64_t modulus = 100;
  uint64_t length = 1024;

  AlignedVector64<uint64_t> values =
      GenerateInsecureUniformRandomValues(length, modulus);
  EXPECT_EQ(values.size(), length);
  EXPECT_TRUE(std::all_of(values.begin(), values.end(),
                          [&](uint64_t x) { return x < modulus; }));
  EXPECT_TRUE(std::any_of(values.begin(), values.end(),
                          [&](uint64_t x) { return x = modulus - 1; }));
}

}  // namespace hexl
}  // namespace intel
