// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"

namespace intel {
namespace hexl {

// Checks whether x == y.
inline void CheckEqual(const std::vector<uint64_t>& x,
                       const std::vector<uint64_t>& y) {
  EXPECT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(x[i], y[i]) << "Mismatch at index " << i;
  }
}

// Asserts x == y
template <typename T>
inline void AssertEqual(const std::vector<T>& x, const std::vector<T>& y) {
  ASSERT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], y[i]) << "Mismatch at index " << i;
  }
}

}  // namespace hexl
}  // namespace intel
