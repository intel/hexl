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
#endif

}  // namespace hexl
}  // namespace intel
