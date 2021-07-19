// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "eltwise/eltwise-cmp-sub-mod-avx512.hpp"
#include "eltwise/eltwise-cmp-sub-mod-internal.hpp"
#include "hexl/eltwise/eltwise-cmp-sub-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util-avx512.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

// Checks AVX512 and native implementations match
#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseCmpSubMod, AVX512) {
  if (!has_avx512dq) {
    return;
  }

  uint64_t length = 172;
  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t cmp = 0; cmp < 8; ++cmp) {
    for (size_t bits = 48; bits <= 51; ++bits) {
      uint64_t modulus = GeneratePrimes(1, bits, 1024)[0];
      std::uniform_int_distribution<uint64_t> distrib(0, modulus - 1);

      for (size_t trial = 0; trial < 200; ++trial) {
        std::vector<uint64_t> op1(length, 0);
        uint64_t bound = distrib(gen);
        uint64_t diff = distrib(gen);
        std::vector<uint64_t> op3(length, 0);
        for (size_t i = 0; i < length; ++i) {
          op1[i] = distrib(gen);
          op3[i] = distrib(gen);
        }
        std::vector<uint64_t> op1a = op1;
        std::vector<uint64_t> op1b = op1;
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
