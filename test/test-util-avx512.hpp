// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "test-util.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
// Checks that at each index, the packed 64-bit integer values in a and b are
// within a difference of at most tolerance
inline void CheckClose(const __m512i a, const __m512i b, uint64_t tolerance) {
  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);
  CheckClose(as, bs, tolerance);
}

// Checks that at each index, the packed 64-bit integer values in a and b match
inline void CheckEqual(const __m512i a, const __m512i b) {
  CheckClose(a, b, 0);
}

// Asserts that at each index, the packed 64-bit integer values in a and b are
// within a difference of at most tolerance
inline void AssertClose(const __m512i a, const __m512i b, uint64_t tolerance) {
  std::vector<uint64_t> as = ExtractValues(a);
  std::vector<uint64_t> bs = ExtractValues(b);
  AssertClose(as, bs, tolerance);
}

// Asserts that at each index, the packed 64-bit integer values in a and b match
inline void AssertEqual(const __m512i a, const __m512i b) {
  AssertClose(a, b, 0);
}
#endif

}  // namespace hexl
}  // namespace intel
