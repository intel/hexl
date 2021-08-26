// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <limits>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"

namespace intel {
namespace hexl {

// Checks whether x and y are within tolerance
inline void CheckClose(const std::vector<uint64_t>& x,
                       const std::vector<uint64_t>& y, uint64_t tolerance) {
  EXPECT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    EXPECT_LE(std::max(x[i], y[i]) - std::min(x[i], y[i]), tolerance)
        << "Mismatch at index " << i;
  }
}

inline void CheckEqual(const std::vector<uint64_t>& x,
                       const std::vector<uint64_t>& y) {
  CheckClose(x, y, 0);
}

// Asserts x and y are within tolerance
template <typename T>
inline void AssertClose(const std::vector<T>& x, const std::vector<T>& y,
                        uint64_t tolerance) {
  ASSERT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    ASSERT_LE(std::max(x[i], y[i]) - std::min(x[i], y[i]), tolerance)
        << "Mismatch at index " << i;
  }
}

template <typename T>
inline void AssertEqual(const std::vector<T>& x, const std::vector<T>& y) {
  AssertClose(x, y, 0);
}

}  // namespace hexl
}  // namespace intel
