// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-sub-mod-internal.hpp"
#include "hexl/eltwise/eltwise-sub-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(EltwiseSubMod, vector_vector_bad_input) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 9, 2, 4, 6};
  std::vector<uint64_t> big_input{11, 12, 13, 14, 15, 16, 17, 18};
  uint64_t modulus = 10;

  EXPECT_ANY_THROW(
      EltwiseSubMod(nullptr, op1.data(), op2.data(), op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseSubMod(op1.data(), nullptr, op2.data(), op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseSubMod(op1.data(), op1.data(), nullptr, op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseSubMod(op1.data(), op1.data(), op2.data(), 0, modulus));
  EXPECT_ANY_THROW(
      EltwiseSubMod(op1.data(), op1.data(), op2.data(), op1.size(), 1));
  EXPECT_ANY_THROW(EltwiseSubMod(op1.data(), big_input.data(), op2.data(),
                                 op1.size(), modulus));
  EXPECT_ANY_THROW(EltwiseSubMod(op1.data(), op1.data(), big_input.data(),
                                 op1.size(), modulus));
}

TEST(EltwiseSubMod, vector_scalar_bad_input) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t op2{1};
  std::vector<uint64_t> big_input{11, 12, 13, 14, 15, 16, 17, 18};
  uint64_t modulus = 10;

  EXPECT_ANY_THROW(
      EltwiseSubMod(nullptr, op1.data(), op2, op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseSubMod(op1.data(), nullptr, op2, op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseSubMod(op1.data(), op1.data(), modulus, op1.size(), modulus));
  EXPECT_ANY_THROW(EltwiseSubMod(op1.data(), op1.data(), op2, 0, modulus));
  EXPECT_ANY_THROW(EltwiseSubMod(op1.data(), op1.data(), op2, op1.size(), 1));
  EXPECT_ANY_THROW(
      EltwiseSubMod(op1.data(), big_input.data(), op2, op1.size(), modulus));
}
#endif

TEST(EltwiseSubMod, vector_vector_native_small) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 9, 4, 4, 6};
  std::vector<uint64_t> exp_out{0, 9, 8, 7, 6, 2, 3, 2};
  uint64_t modulus = 10;

  EltwiseSubModNative(op1.data(), op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseSubMod, vector_scalar_native_small) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t op2{3};
  std::vector<uint64_t> exp_out{8, 9, 0, 1, 2, 3, 4, 5};
  uint64_t modulus = 10;

  EltwiseSubModNative(op1.data(), op1.data(), op2, op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseSubMod, vector_vector_native_big) {
  uint64_t modulus = GeneratePrimes(1, 60, true, 1024)[0];

  std::vector<uint64_t> op1{0,           1,           2,           3,
                            modulus - 1, modulus - 2, modulus - 3, modulus - 4};
  std::vector<uint64_t> op2{modulus - 1, modulus - 2, 3, 2,
                            modulus - 3, modulus - 4, 1, 0};
  std::vector<uint64_t> exp_out{1, 3, modulus - 1, 1,
                                2, 2, modulus - 4, modulus - 4};

  EltwiseSubModNative(op1.data(), op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseSubMod, vector_scalar_native_big) {
  uint64_t modulus = GeneratePrimes(1, 60, true, 1024)[0];

  std::vector<uint64_t> op1{0,           1,           2,           3,
                            modulus - 1, modulus - 2, modulus - 3, modulus - 4};
  uint64_t op2{modulus - 1};
  std::vector<uint64_t> exp_out{1, 2,           3,           4,
                                0, modulus - 1, modulus - 2, modulus - 3};

  EltwiseSubModNative(op1.data(), op1.data(), op2, op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

}  // namespace hexl
}  // namespace intel
