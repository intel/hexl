// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-cmp-sub-mod-avx512.hpp"
#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "hexl/eltwise/eltwise-cmp-sub-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util-avx512.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

// Checks AVX512 and native implementations match
#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseCmpSubMod, AVX512) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  uint64_t length = 172;

  for (size_t cmp = 0; cmp < 8; ++cmp) {
    for (size_t bits = 48; bits <= 51; ++bits) {
      uint64_t modulus = GeneratePrimes(1, bits, true, 1024)[0];

      for (size_t trial = 0; trial < 200; ++trial) {
        auto op1 = GenerateInsecureUniformRandomValues(length, 0, modulus);
        auto op3 = GenerateInsecureUniformRandomValues(length, 0, modulus);

        uint64_t bound = GenerateInsecureUniformRandomValue(0, modulus);
        // Ensure diff != 0
        uint64_t diff = GenerateInsecureUniformRandomValue(1, modulus - 1);

        auto op1a = op1;
        auto op1b = op1;
        std::vector<uint64_t> op1_out(op1.size(), 0);
        std::vector<uint64_t> op1a_out(op1.size(), 0);
        std::vector<uint64_t> op1b_out(op1.size(), 0);

        EltwiseCmpSubMod(op1_out.data(), op1.data(), op1.size(), modulus,
                         static_cast<CMPINT>(cmp), bound, diff);
        EltwiseCmpSubModNative(op1a_out.data(), op1a.data(), op1a.size(),
                               modulus, static_cast<CMPINT>(cmp), bound, diff);
        EltwiseCmpSubModAVX512(op1b_out.data(), op1b.data(), op1b.size(),
                               modulus, static_cast<CMPINT>(cmp), bound, diff);

        ASSERT_EQ(op1_out, op1a_out);
        ASSERT_EQ(op1_out, op1b_out);
      }
    }
  }
}
#endif
}  // namespace hexl
}  // namespace intel
