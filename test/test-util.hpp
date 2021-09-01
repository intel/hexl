// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"

namespace intel {
namespace hexl {

// Generates a vector of size random values drawn uniformly from [0, modulus)
inline AlignedVector64<uint64_t> GenerateUniformRandomValues(uint64_t size,
                                                             uint64_t modulus) {
  AlignedVector64<uint64_t> values(size);
  std::random_device rd;
  std::mt19937 mersenne_engine(rd());
  std::uniform_int_distribution<uint64_t> distrib(0, modulus - 1);

  auto generator = [&distrib, &mersenne_engine]() {
    return distrib(mersenne_engine);
  };

  std::generate(values.begin(), values.end(), generator);
  return values;
}

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
inline void AssertClose(const T& x, const T& y, uint64_t tolerance) {
  ASSERT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    ASSERT_LE(std::max(x[i], y[i]) - std::min(x[i], y[i]), tolerance)
        << "Mismatch at index " << i;
  }
}

template <typename T>
inline void AssertEqual(const T& x, const T& y) {
  AssertClose(x, y, 0);
}

}  // namespace hexl
}  // namespace intel
