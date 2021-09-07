// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

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
  uint64_t modulus = GeneratePrimes(1, 51, true, 1024)[0];

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
  uint64_t modulus = GeneratePrimes(1, 60, true, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 4, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModNative<1>(result.data(), op1.data(), op2.data(), op1.size(),
                          modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, 8big) {
  uint64_t modulus = GeneratePrimes(1, 48, true, 1024)[0];

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
  uint64_t modulus = GeneratePrimes(1, 51, true, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{modulus - 4, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 8, 14, 18, 20, 20, 18, 14, 8};

  EltwiseMultMod(result.data(), op1.data(), op2.data(), op1.size(), modulus, 1);

  CheckEqual(result, exp_out);
}

}  // namespace hexl
}  // namespace intel
