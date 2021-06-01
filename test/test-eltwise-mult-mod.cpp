// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(EltwiseMultMod, null) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t modulus = 769;
  std::vector<uint64_t> big_input(op1.size(), modulus);

  EXPECT_ANY_THROW(
      EltwiseMultMod(nullptr, op1.data(), op2.data(), op1.size(), modulus, 1));
  EXPECT_ANY_THROW(
      EltwiseMultMod(op1.data(), nullptr, op2.data(), op1.size(), modulus, 1));
  EXPECT_ANY_THROW(
      EltwiseMultMod(op1.data(), op1.data(), nullptr, op1.size(), modulus, 1));
  EXPECT_ANY_THROW(
      EltwiseMultMod(op1.data(), op1.data(), op2.data(), 0, modulus, 1));
  EXPECT_ANY_THROW(
      EltwiseMultMod(op1.data(), op1.data(), op2.data(), op1.size(), 1, 1));
  EXPECT_ANY_THROW(EltwiseMultMod(op1.data(), op1.data(), op2.data(),
                                  op1.size(), modulus, 0));
  EXPECT_ANY_THROW(EltwiseMultMod(op1.data(), big_input.data(), op2.data(),
                                  op1.size(), modulus, 1));
  EXPECT_ANY_THROW(EltwiseMultMod(op1.data(), op1.data(), big_input.data(),
                                  op1.size(), modulus, 1));
}
#endif

TEST(EltwiseMultModInPlace, 4) {
  std::vector<uint64_t> op1{2, 4, 3, 2};
  std::vector<uint64_t> op2{2, 1, 2, 0};
  std::vector<uint64_t> exp_out{4, 4, 6, 0};

  uint64_t modulus = 769;

  EltwiseMultMod(op1.data(), op1.data(), op2.data(), op1.size(), modulus, 1);
  CheckEqual(op1, exp_out);
}

TEST(EltwiseMultModInPlace, 6) {
  std::vector<uint64_t> op1{0, 1, 2, 3, 4, 5};
  std::vector<uint64_t> op2{2, 4, 6, 8, 10, 12};
  std::vector<uint64_t> exp_out{0, 4, 12, 24, 40, 60};

  uint64_t modulus = 769;

  EltwiseMultMod(op1.data(), op1.data(), op2.data(), op1.size(), modulus, 1);
  CheckEqual(op1, exp_out);
}

#ifdef HEXL_DEBUG
TEST(EltwiseMultModInPlace, 8_bounds) {
  std::vector<uint64_t> op1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint64_t> op2{0, 1, 2, 3, 4, 5, 6, 770};

  uint64_t modulus = 769;

  EXPECT_ANY_THROW(EltwiseMultMod(op1.data(), op1.data(), op2.data(),
                                  op1.size(), modulus, 1));
}
#endif

TEST(EltwiseMultModInPlace, 9) {
  uint64_t modulus = GeneratePrimes(1, 51, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{modulus - 4, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<uint64_t> exp_out{12, 8, 14, 18, 20, 20, 18, 14, 8};

  EltwiseMultMod(op1.data(), op1.data(), op2.data(), op1.size(), modulus, 1);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseMultMod, native_mult2) {
  std::vector<uint64_t> op1{1, 2,  3,  4,  5,  6,  7,  8,
                            9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<uint64_t> op2{17, 18, 19, 20, 21, 22, 23, 24,
                            25, 26, 27, 28, 29, 30, 31, 32};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0};
  std::vector<uint64_t> exp_out{17, 36, 57, 80, 4,  31, 60, 91,
                                23, 58, 95, 33, 74, 16, 61, 7};
  uint64_t modulus = 101;

  EltwiseMultModNative<1>(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, native2_big) {
  uint64_t modulus = GeneratePrimes(1, 60, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 4, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative<1>(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, 8big) {
  uint64_t modulus = GeneratePrimes(1, 48, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{1, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative<1>(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, 8big2) {
  uint64_t modulus = 281474976749569;

  std::vector<uint64_t> op1{(modulus - 1) / 2, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{(modulus + 1) / 2, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{70368744187392, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative<1>(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, 8big3) {
  uint64_t modulus = 1125891450734593;

  std::vector<uint64_t> op1{1078888294739028, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{1114802337613200, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{13344071208410, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative<1>(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}
#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseMultMod, avx512_small) {
  std::vector<uint64_t> op1{1, 2, 3, 1, 1, 1, 0, 1, 0};
  std::vector<uint64_t> op2{1, 1, 1, 1, 2, 3, 1, 0, 0};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{1, 2, 3, 1, 2, 3, 0, 0, 0};

  uint64_t modulus = 769;
  EltwiseMultModAVX512Float<1>(result.data(), op1.data(), op2.data(),
                               op1.size(), modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, avx512_int2) {
  uint64_t modulus = GeneratePrimes(1, 60, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 4, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModAVX512Int<2>(result.data(), op1.data(), op2.data(), op1.size(),
                             modulus);

  CheckEqual(result, exp_out);
}

#endif

TEST(EltwiseMultMod, 4) {
  std::vector<uint64_t> op1{2, 4, 3, 2};
  std::vector<uint64_t> op2{2, 1, 2, 0};
  std::vector<uint64_t> result{0, 0, 0, 0};
  std::vector<uint64_t> exp_out{4, 4, 6, 0};

  uint64_t modulus = 769;

  EltwiseMultMod(result.data(), op1.data(), op2.data(), op1.size(), modulus, 1);
  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, 6) {
  std::vector<uint64_t> op1{0, 1, 2, 3, 4, 5};
  std::vector<uint64_t> op2{2, 4, 6, 8, 10, 12};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{0, 4, 12, 24, 40, 60};

  uint64_t modulus = 769;

  EltwiseMultMod(result.data(), op1.data(), op2.data(), op1.size(), modulus, 1);
  CheckEqual(result, exp_out);
}

#ifdef HEXL_DEBUG
TEST(EltwiseMultMod, 8_bounds) {
  std::vector<uint64_t> op1{0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<uint64_t> op2{0, 1, 2, 3, 4, 5, 6, 770};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 769;

  EXPECT_ANY_THROW(EltwiseMultMod(result.data(), op1.data(), op2.data(),
                                  op1.size(), modulus, 1));
}
#endif

TEST(EltwiseMultMod, 9) {
  uint64_t modulus = GeneratePrimes(1, 51, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{modulus - 4, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 8, 14, 18, 20, 20, 18, 14, 8};

  EltwiseMultMod(result.data(), op1.data(), op2.data(), op1.size(), modulus, 1);

  CheckEqual(result, exp_out);
}

#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseMultMod, Big) {
  uint64_t modulus = 1125891450734593;

  std::vector<uint64_t> op1{706712574074152, 943467560561867, 1115920708919443,
                            515713505356094, 525633777116309, 910766532971356,
                            757086506562426, 799841520990167, 1};
  std::vector<uint64_t> op2{515910833966633, 96924929169117,   537587376997453,
                            41829060600750,  205864998008014,  463185427411646,
                            965818279134294, 1075778049568657, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{
      231838787758587, 618753612121218, 1116345967490421,
      409735411065439, 25680427818594,  950138933882289,
      554128714280822, 1465109636753,   1};

  EltwiseMultModAVX512Int<4>(result.data(), op1.data(), op2.data(), op1.size(),
                             modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, 8192) {
  std::random_device rd;
  std::mt19937 gen(rd());

  size_t length = 8192;

  uint64_t input_mod_factor = 1;
  uint64_t modulus = (1ULL << 53) + 7;
  std::uniform_int_distribution<uint64_t> distrib(
      0, input_mod_factor * modulus - 1);

  std::vector<uint64_t> op1(length, 0);
  std::vector<uint64_t> op2(length, 0);
  std::vector<uint64_t> out_avx(length, 0);
  std::vector<uint64_t> out_native(length, 0);

  for (size_t i = 0; i < length; ++i) {
    op1[i] = distrib(gen);
    op2[i] = distrib(gen);
  }

  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{
      231838787758587, 618753612121218, 1116345967490421,
      409735411065439, 25680427818594,  950138933882289,
      554128714280822, 1465109636753,   1};

  EltwiseMultModAVX512Int<1>(out_avx.data(), op1.data(), op2.data(), op1.size(),
                             modulus);

  EltwiseMultModNative<1>(out_native.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(out_avx, out_native);
}

TEST(EltwiseMultMod, 16384) {
  std::random_device rd;
  std::mt19937 gen(rd());

  size_t length = 16384;

  uint64_t input_mod_factor = 1;
  uint64_t modulus = (1ULL << 53) + 7;
  std::uniform_int_distribution<uint64_t> distrib(
      0, input_mod_factor * modulus - 1);

  std::vector<uint64_t> op1(length, 0);
  std::vector<uint64_t> op2(length, 0);
  std::vector<uint64_t> out_avx(length, 0);
  std::vector<uint64_t> out_native(length, 0);

  for (size_t i = 0; i < length; ++i) {
    op1[i] = distrib(gen);
    op2[i] = distrib(gen);
  }

  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{
      231838787758587, 618753612121218, 1116345967490421,
      409735411065439, 25680427818594,  950138933882289,
      554128714280822, 1465109636753,   1};

  EltwiseMultModAVX512Int<1>(out_avx.data(), op1.data(), op2.data(), op1.size(),
                             modulus);

  EltwiseMultModNative<1>(out_native.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(out_avx, out_native);
}

#endif

// Checks AVX512 and native eltwise mult Out-of-Place implementations match
#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseMultMod, AVX512Big) {
  std::random_device rd;
  std::mt19937 gen(rd());

  size_t length = 173;

  for (size_t input_mod_factor = 1; input_mod_factor <= 4;
       input_mod_factor *= 2) {
    for (size_t bits = 1; bits <= 60; ++bits) {
      uint64_t modulus = (1ULL << bits) + 7;
      std::uniform_int_distribution<uint64_t> distrib(
          0, input_mod_factor * modulus - 1);

      bool use_avx512_float = (input_mod_factor * modulus < MaximumValue(50));

#ifdef HEXL_DEBUG
      size_t num_trials = 10;
#else
      size_t num_trials = 100;
#endif
      for (size_t trial = 0; trial < num_trials; ++trial) {
        std::vector<uint64_t> op1(length, 0);
        std::vector<uint64_t> op2(length, 0);
        std::vector<uint64_t> rs1(length, 0);
        std::vector<uint64_t> rs2(length, 0);
        std::vector<uint64_t> rs3(length, 0);
        std::vector<uint64_t> rs4(length, 0);
        for (size_t i = 0; i < length; ++i) {
          op1[i] = distrib(gen);
          op2[i] = distrib(gen);
        }
        op1[0] = input_mod_factor * modulus - 1;
        op2[0] = input_mod_factor * modulus - 1;

        switch (input_mod_factor) {
          case 1:
            EltwiseMultModNative<1>(rs1.data(), op1.data(), op2.data(),
                                    op1.size(), modulus);
            if (use_avx512_float) {
              EltwiseMultModAVX512Float<1>(rs2.data(), op1.data(), op2.data(),
                                           op1.size(), modulus);
            } else {
              EltwiseMultModAVX512Int<1>(rs3.data(), op1.data(), op2.data(),
                                         op1.size(), modulus);
            }
            break;
          case 2:
            EltwiseMultModNative<2>(rs1.data(), op1.data(), op2.data(),
                                    op1.size(), modulus);
            if (use_avx512_float) {
              EltwiseMultModAVX512Float<2>(rs2.data(), op1.data(), op2.data(),
                                           op1.size(), modulus);
            } else {
              EltwiseMultModAVX512Int<2>(rs3.data(), op1.data(), op2.data(),
                                         op1.size(), modulus);
            }
            break;
          case 4:
            EltwiseMultModNative<4>(rs1.data(), op1.data(), op2.data(),
                                    op1.size(), modulus);
            if (use_avx512_float) {
              EltwiseMultModAVX512Float<4>(rs2.data(), op1.data(), op2.data(),
                                           op1.size(), modulus);
            } else {
              EltwiseMultModAVX512Int<4>(rs3.data(), op1.data(), op2.data(),
                                         op1.size(), modulus);
            }
            break;
        }
        EltwiseMultMod(rs4.data(), op1.data(), op2.data(), op1.size(), modulus,
                       input_mod_factor);

        ASSERT_EQ(rs4, rs1);

        ASSERT_EQ(rs1[0], 1);
        if (use_avx512_float) {
          ASSERT_EQ(rs1, rs2);
          ASSERT_EQ(rs2[0], 1);
        } else {
          ASSERT_EQ(rs1, rs3);
          ASSERT_EQ(rs3[0], 1);
        }
      }
    }
  }
}
#endif
}  // namespace hexl
}  // namespace intel
