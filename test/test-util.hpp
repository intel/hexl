// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <vector>

#include "logging/logging.hpp"
#include "util/avx512-util.hpp"
#include "util/check.hpp"

namespace intel {
namespace hexl {

// Checks whether x == y.
inline void CheckEqual(const std::vector<uint64_t>& x,
                       const std::vector<uint64_t>& y) {
  EXPECT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    EXPECT_EQ(x[i], y[i]);
  }
}

// Asserts x == y
template <typename T>
inline void AssertEqual(const std::vector<T>& x, const std::vector<T>& y) {
  ASSERT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], y[i]);
  }
}

#ifdef HEXL_HAS_AVX512DQ
inline void CheckEqual(const __m512i a, const __m512i b) {
  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);
  CheckEqual(as, bs);
}

inline void AssertEqual(const __m512i a, const __m512i b) {
  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);
  AssertEqual(as, bs);
}

// Returns true iff a == b
// Logs an error if a != b
inline bool Equals(__m512i a, __m512i b) {
  bool match = true;

  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);

  for (size_t i = 0; i < 8; ++i) {
    if (as[i] != bs[i]) {
      std::cerr << "Mismatch at index " << i << ": "
                << "a[" << i << "] = " << as[i] << ", b[" << i
                << "] = " << bs[i] << "\n";
      match = false;
    }
  }
  return match;
}
#endif

}  // namespace hexl
}  // namespace intel
