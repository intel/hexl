// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-cmp-add-avx512.hpp"
#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "hexl/eltwise/eltwise-cmp-add.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util-avx512.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

// Checks AVX512 and native implementations match
#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseCmpAdd, AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  uint64_t length = 1025;
  uint64_t modulus = 100;

  for (size_t cmp = 0; cmp < 8; ++cmp) {
    for (size_t trial = 0; trial < 200; ++trial) {
      auto op1 = GenerateInsecureUniformRandomValues(length, 0, modulus);
      uint64_t bound = GenerateInsecureUniformRandomValue(0, modulus);
      uint64_t diff = GenerateInsecureUniformRandomValue(1, modulus);

      auto op1a = op1;
      auto op1b = op1;
      AlignedVector64<uint64_t> op1_out(op1.size(), 0);
      AlignedVector64<uint64_t> op1a_out(op1.size(), 0);
      AlignedVector64<uint64_t> op1b_out(op1.size(), 0);

      EltwiseCmpAdd(op1_out.data(), op1.data(), op1.size(),
                    static_cast<CMPINT>(cmp), bound, diff);
      EltwiseCmpAddNative(op1a_out.data(), op1a.data(), op1a.size(),
                          static_cast<CMPINT>(cmp), bound, diff);
      EltwiseCmpAddAVX512(op1b_out.data(), op1b.data(), op1b.size(),
                          static_cast<CMPINT>(cmp), bound, diff);

      ASSERT_EQ(op1_out, op1a_out);
      ASSERT_EQ(op1_out, op1b_out);
    }
  }
}
#endif

}  // namespace hexl
}  // namespace intel
