// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-fma-mod-avx512.hpp"
#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "hexl/eltwise/eltwise-fma-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util-avx512.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseFMAMod, avx512_small) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 2;
  std::vector<uint64_t> arg3{1, 1, 1, 1, 2, 3, 1, 0};
  std::vector<uint64_t> exp_out{3, 5, 7, 9, 12, 15, 15, 16};

  uint64_t modulus = 101;
  EltwiseFMAModAVX512<64, 1>(arg1.data(), arg1.data(), arg2, arg3.data(),
                             arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, avx512_small2) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> exp_out{26, 44, 62, 80, 98, 15, 33, 51};

  uint64_t modulus = 101;

  EltwiseFMAModAVX512<64, 1>(arg1.data(), arg1.data(), arg2, arg3.data(),
                             arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, avx512_mult1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> arg1{1, 2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24,
                             25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{34, 52, 70, 88, 5,  23, 41, 59,
                                77, 95, 12, 30, 48, 66, 84, 1};

  uint64_t modulus = 101;

  EltwiseFMAModAVX512<64, 1>(arg1.data(), arg1.data(), arg2, arg3.data(),
                             arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, avx512_mult2) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> arg1{102, 2,  3,  4,  5,  6,  7,  8,
                             9,   10, 11, 12, 13, 14, 15, 16};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24,
                             25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{34, 52, 70, 88, 5,  23, 41, 59,
                                77, 95, 12, 30, 48, 66, 84, 1};

  uint64_t modulus = 101;

  EltwiseFMAModAVX512<64, 2>(arg1.data(), arg1.data(), arg2, arg3.data(),
                             arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, avx512_mult4) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> arg1{400, 2,  3,  4,  5,  6,  7,  8,
                             9,   10, 11, 12, 13, 14, 15, 16};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24,
                             25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{50, 52, 70, 88, 5,  23, 41, 59,
                                77, 95, 12, 30, 48, 66, 84, 1};

  uint64_t modulus = 101;

  EltwiseFMAModAVX512<64, 4>(arg1.data(), arg1.data(), arg2, arg3.data(),
                             arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, avx512_mult8) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> arg1{800, 2,  3,  4,  5,  6,  7,  8,
                             9,   10, 11, 12, 13, 14, 15, 16};
  uint64_t arg2 = 17;
  std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24,
                             25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> exp_out{83, 52, 70, 88, 5,  23, 41, 59,
                                77, 95, 12, 30, 48, 66, 84, 1};

  uint64_t modulus = 101;

  EltwiseFMAModAVX512<64, 8>(arg1.data(), arg1.data(), arg2, arg3.data(),
                             arg1.size(), modulus);

  CheckEqual(arg1, exp_out);
}
#endif

// Check AVX512DQ and native eltwise FMA implementations match
#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseFMAMod, AVX512DQ) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  uint64_t length = 1031;

  for (size_t input_mod_factor = 1; input_mod_factor <= 8;
       input_mod_factor *= 2) {
    for (size_t bits = 1; bits <= 60; ++bits) {
      uint64_t modulus = (1ULL << bits) + 7;

#ifdef HEXL_DEBUG
      size_t num_trials = 10;
#else
      size_t num_trials = 100;
#endif

      for (size_t trial = 0; trial < num_trials; ++trial) {
        auto arg1 = GenerateInsecureUniformRandomValues(
            length, 0, input_mod_factor * modulus);
        uint64_t arg2 =
            GenerateInsecureUniformRandomValue(0, input_mod_factor * modulus);
        auto arg3 = GenerateInsecureUniformRandomValues(
            length, 0, input_mod_factor * modulus);

        std::vector<uint64_t> out_default(length, 0);
        std::vector<uint64_t> out_native(length, 0);
        std::vector<uint64_t> out_avx(length, 0);

        uint64_t* arg3_data = (trial % 2 == 0) ? arg3.data() : nullptr;

        EltwiseFMAMod(out_default.data(), arg1.data(), arg2, arg3_data,
                      arg1.size(), modulus, input_mod_factor);

        switch (input_mod_factor) {
          case 1:
            EltwiseFMAModNative<1>(out_native.data(), arg1.data(), arg2,
                                   arg3_data, arg1.size(), modulus);
            EltwiseFMAModAVX512<64, 1>(out_avx.data(), arg1.data(), arg2,
                                       arg3_data, arg1.size(), modulus);
            break;
          case 2:
            EltwiseFMAModNative<2>(out_native.data(), arg1.data(), arg2,
                                   arg3_data, arg1.size(), modulus);
            EltwiseFMAModAVX512<64, 2>(out_avx.data(), arg1.data(), arg2,
                                       arg3_data, arg1.size(), modulus);
            break;
          case 4:
            EltwiseFMAModNative<4>(out_native.data(), arg1.data(), arg2,
                                   arg3_data, arg1.size(), modulus);
            EltwiseFMAModAVX512<64, 4>(out_avx.data(), arg1.data(), arg2,
                                       arg3_data, arg1.size(), modulus);
            break;
          case 8:
            EltwiseFMAModNative<8>(out_native.data(), arg1.data(), arg2,
                                   arg3_data, arg1.size(), modulus);
            EltwiseFMAModAVX512<64, 8>(out_avx.data(), arg1.data(), arg2,
                                       arg3_data, arg1.size(), modulus);
            break;
        }

        ASSERT_EQ(out_default, out_native);
        ASSERT_EQ(out_default, out_avx);
      }
    }
  }
}
#endif

// Checks AVX512IFMA and native eltwise FMA implementations match
#ifdef HEXL_HAS_AVX512IFMA
TEST(EltwiseFMAMod, AVX512IFMA) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }

  uint64_t length = 1024;

  constexpr uint64_t input_mod_factor = 8;

  for (size_t bits = 48; bits <= 51; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];
    for (size_t trial = 0; trial < 1000; ++trial) {
      auto arg1 = GenerateInsecureUniformRandomValues(
          length, 0, input_mod_factor * modulus);
      uint64_t arg2 = GenerateInsecureUniformRandomValue(0, modulus);
      auto arg3 = GenerateInsecureUniformRandomValues(
          length, 0, input_mod_factor * modulus);

      auto arg1a = arg1;
      auto arg1b = arg1;

      uint64_t* arg3_data = (trial % 2 == 0) ? arg3.data() : nullptr;

      EltwiseFMAMod(arg1.data(), arg1.data(), arg2, arg3_data, arg1.size(),
                    modulus, input_mod_factor);

      if (has_avx512ifma) {
        EltwiseFMAModAVX512<52, input_mod_factor>(
            arg1a.data(), arg1a.data(), arg2, arg3_data, arg1.size(), modulus);
        ASSERT_EQ(arg1, arg1a);
      }

      EltwiseFMAModAVX512<64, input_mod_factor>(
          arg1b.data(), arg1b.data(), arg2, arg3_data, arg1.size(), modulus);

      ASSERT_EQ(arg1, arg1b);
    }
  }
}
#endif

}  // namespace hexl
}  // namespace intel
