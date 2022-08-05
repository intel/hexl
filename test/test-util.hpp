// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <complex>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"

namespace intel {
namespace hexl {

// Checks whether x and y are within tolerance
template <typename A, typename B>
inline void CheckClose(const A& x, const B& y, double tolerance) {
  EXPECT_EQ(x.size(), y.size());
  uint64_t N = std::min(x.size(), y.size());
  for (size_t i = 0; i < N; ++i) {
    EXPECT_LE(std::max(x[i], y[i]) - std::min(x[i], y[i]), tolerance)
        << "Mismatch at index " << i;
  }
}

// Vector of complex
template <>
inline void CheckClose(const AlignedVector64<std::complex<double>>& x,
                       const AlignedVector64<std::complex<double>>& y,
                       double tolerance) {
  EXPECT_EQ(x.size(), y.size());
  uint64_t N = std::min(x.size(), y.size());
  for (size_t i = 0; i < N; ++i) {
    EXPECT_LE(
        std::max(x[i].real(), y[i].real()) - std::min(x[i].real(), y[i].real()),
        tolerance)
        << "Mismatch at (real part) index " << i;
    EXPECT_LE(
        std::max(x[i].imag(), y[i].imag()) - std::min(x[i].imag(), y[i].imag()),
        tolerance)
        << "Mismatch at (imaginary part) index " << i;
  }
}

// Single complex value
template <>
inline void CheckClose(const std::complex<double>& x,
                       const std::complex<double>& y, double tolerance) {
  EXPECT_LE(std::max(x.real(), y.real()) - std::min(x.real(), y.real()),
            tolerance)
      << "Mismatch at real value";
  EXPECT_LE(std::max(x.imag(), y.imag()) - std::min(x.imag(), y.imag()),
            tolerance)
      << "Mismatch at imaginary value ";
}

inline void CheckEqual(const std::vector<uint64_t>& x,
                       const std::vector<uint64_t>& y) {
  CheckClose(x, y, 0);
}

// Asserts x and y are within tolerance
template <typename A, typename B>
inline void AssertClose(const A& x, const B& y, uint64_t tolerance) {
  ASSERT_EQ(x.size(), y.size());
  uint64_t N = x.size();
  for (size_t i = 0; i < N; ++i) {
    ASSERT_LE(std::max(x[i], y[i]) - std::min(x[i], y[i]), tolerance)
        << "Mismatch at index " << i;
  }
}

template <typename A, typename B>
inline void AssertEqual(const A& x, const B& y) {
  AssertClose(x, y, 0);
}

}  // namespace hexl
}  // namespace intel
