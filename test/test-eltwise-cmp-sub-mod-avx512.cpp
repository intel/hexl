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
#ifdef HEXL_HAS_AVX512IFMA
TEST(EltwiseCmpSubMod, AVX512_52) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  uint64_t length = 9;
  uint64_t modulus = 1125896819525633;

  for (size_t trial = 0; trial < 200; ++trial) {
    auto op1 = std::vector<uint64_t>(length, 1106601337915084531);
    uint64_t bound = 576460751967876096;
    uint64_t diff = 3160741504001;

    auto op1_native = op1;
    auto op1_avx512 = op1;
    std::vector<uint64_t> op1_out(op1.size(), 0);
    std::vector<uint64_t> op1_native_out(op1.size(), 0);
    std::vector<uint64_t> op1_avx512_out(op1.size(), 0);

    EltwiseCmpSubMod(op1_out.data(), op1.data(), op1.size(), modulus,
                     intel::hexl::CMPINT::NLE, bound, diff);
    EltwiseCmpSubModNative(op1_native_out.data(), op1.data(), op1.size(),
                           modulus, intel::hexl::CMPINT::NLE, bound, diff);
    EltwiseCmpSubModAVX512<52>(op1_avx512_out.data(), op1.data(), op1.size(),
                               modulus, intel::hexl::CMPINT::NLE, bound, diff);

    ASSERT_EQ(op1_out, op1_native_out);
    ASSERT_EQ(op1_native_out, op1_avx512_out);
  }
}
#endif

#ifdef HEXL_HAS_AVX512IFMA
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
        EltwiseCmpSubModAVX512<52>(op1b_out.data(), op1b.data(), op1b.size(),
                                   modulus, static_cast<CMPINT>(cmp), bound,
                                   diff);

        ASSERT_EQ(op1_out, op1a_out);
        ASSERT_EQ(op1_out, op1b_out);
      }
    }
  }
}

TEST(EltwiseCmpSubMod, AVX512_64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  uint64_t length = 9;
  uint64_t modulus = 1152921504606748673;

  for (size_t trial = 0; trial < 200; ++trial) {
    auto op1 = std::vector<uint64_t>(length, 64961);
    uint64_t bound = 576460752303415296;
    uint64_t diff = 81920;

    auto op1_native = op1;
    auto op1_avx512 = op1;
    std::vector<uint64_t> op1_out(op1.size(), 0);
    std::vector<uint64_t> op1_native_out(op1.size(), 0);
    std::vector<uint64_t> op1_avx512_out(op1.size(), 0);

    EltwiseCmpSubMod(op1_out.data(), op1.data(), op1.size(), modulus,
                     intel::hexl::CMPINT::NLE, bound, diff);
    EltwiseCmpSubModNative(op1_native_out.data(), op1.data(), op1.size(),
                           modulus, intel::hexl::CMPINT::NLE, bound, diff);
    EltwiseCmpSubModAVX512<64>(op1_avx512_out.data(), op1.data(), op1.size(),
                               modulus, intel::hexl::CMPINT::NLE, bound, diff);

    ASSERT_EQ(op1_out, op1_native_out);
    ASSERT_EQ(op1_native_out, op1_avx512_out);
  }
}

#endif
}  // namespace hexl
}  // namespace intel
