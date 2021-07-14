// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "eltwise/eltwise-fma-mod-avx512.hpp"
#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "hexl/eltwise/eltwise-fma-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(EltwiseFMAMod, null) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};

  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 1;
  std::vector<uint64_t> arg3{9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> exp_out{10, 12, 14, 16, 18, 20, 22, 24};
  uint64_t modulus = 769;
  std::vector<uint64_t> big_input(op1.size(), modulus);

  EXPECT_ANY_THROW(EltwiseFMAMod(nullptr, arg1.data(), arg2, arg3.data(),
                                 arg1.size(), modulus, 1));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), nullptr, arg2, arg3.data(),
                                 arg1.size(), modulus, 1));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), arg1.data(), arg2, arg3.data(), 0,
                                 modulus, 1));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), arg1.data(), arg2, arg3.data(),
                                 arg1.size(), 1, 1));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), arg1.data(), arg2, arg3.data(),
                                 arg1.size(), 1, 99));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), big_input.data(), arg2,
                                 arg3.data(), arg1.size(), modulus, 1));
  EXPECT_ANY_THROW(EltwiseFMAMod(arg1.data(), arg1.data(), arg2,
                                 big_input.data(), arg1.size(), modulus, 1));
}
#endif

TEST(EltwiseFMAMod, small) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t arg2 = 1;
  std::vector<uint64_t> arg3{9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> exp_out{10, 12, 14, 16, 18, 20, 22, 24};
  uint64_t modulus = 769;

  EltwiseFMAMod(arg1.data(), arg1.data(), arg2, arg3.data(), arg1.size(),
                modulus, 1);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, native_null) {
  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t arg2 = 1;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t modulus = 769;

  EltwiseFMAMod(arg1.data(), arg1.data(), arg2, nullptr, arg1.size(), modulus,
                1);

  CheckEqual(arg1, exp_out);
}

TEST(EltwiseFMAMod, mult_input_mod_factor) {
  uint64_t modulus = 101;

  for (uint64_t input_mod_factor = 1; input_mod_factor <= 8;
       input_mod_factor *= 2) {
    uint64_t arg1_add = (input_mod_factor - 1) * modulus;
    std::vector<uint64_t> arg1{arg1_add + 1,  arg1_add + 2,  arg1_add + 3,
                               arg1_add + 4,  arg1_add + 5,  arg1_add + 6,
                               arg1_add + 7,  arg1_add + 8,  arg1_add + 9,
                               arg1_add + 10, arg1_add + 11, arg1_add + 12,
                               arg1_add + 13, arg1_add + 14, arg1_add + 15,
                               arg1_add + 16, arg1_add + 17};

    uint64_t arg2 = 72;
    std::vector<uint64_t> arg3{17, 18, 19, 20, 21, 22, 23, 24, 25,
                               26, 27, 28, 29, 30, 31, 32, 33};
    std::vector<uint64_t> exp_out{89, 61, 33, 5,  78, 50, 22, 95, 67,
                                  39, 11, 84, 56, 28, 0,  73, 45};

    EltwiseFMAMod(arg1.data(), arg1.data(), arg2, arg3.data(), arg1.size(),
                  modulus, input_mod_factor);

    CheckEqual(arg1, exp_out);
  }
}

#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseFMAMod, avx512_small) {
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
  uint64_t length = 1031;
  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t input_mod_factor = 1; input_mod_factor <= 8;
       input_mod_factor *= 2) {
    for (size_t bits = 1; bits <= 60; ++bits) {
      uint64_t modulus = (1ULL << bits) + 7;
      std::uniform_int_distribution<uint64_t> distrib(
          0, input_mod_factor * modulus - 1);

#ifdef HEXL_DEBUG
      size_t num_trials = 10;
#else
      size_t num_trials = 100;
#endif

      for (size_t trial = 0; trial < num_trials; ++trial) {
        std::vector<uint64_t> arg1(length, 0);
        uint64_t arg2 = distrib(gen);
        std::vector<uint64_t> arg3(length, 0);
        for (size_t i = 0; i < length; ++i) {
          arg1[i] = distrib(gen);
          arg3[i] = distrib(gen);
        }
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
TEST(EltwiseFMAMod, AVX512) {
  uint64_t length = 1024;
  std::random_device rd;
  std::mt19937 gen(rd());

  constexpr uint64_t input_mod_factor = 8;

  for (size_t bits = 48; bits <= 51; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, length)[0];
    std::uniform_int_distribution<uint64_t> distrib(
        0, input_mod_factor * modulus - 1);

    for (size_t trial = 0; trial < 1000; ++trial) {
      std::vector<uint64_t> arg1(length, 0);
      uint64_t arg2 = distrib(gen) % modulus;
      std::vector<uint64_t> arg3(length, 0);
      for (size_t i = 0; i < length; ++i) {
        arg1[i] = distrib(gen);
        arg3[i] = distrib(gen);
      }
      std::vector<uint64_t> arg1a = arg1;
      std::vector<uint64_t> arg1b = arg1;

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
