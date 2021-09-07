// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-add-mod-internal.hpp"
#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(EltwiseAddMod, vector_vector_bad_input) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 9, 2, 4, 6};
  std::vector<uint64_t> big_input{11, 12, 13, 14, 15, 16, 17, 18};
  uint64_t modulus = 10;

  EXPECT_ANY_THROW(
      EltwiseAddMod(nullptr, op1.data(), op2.data(), op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseAddMod(op1.data(), nullptr, op2.data(), op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseAddMod(op1.data(), op1.data(), nullptr, op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseAddMod(op1.data(), op1.data(), op2.data(), 0, modulus));
  EXPECT_ANY_THROW(
      EltwiseAddMod(op1.data(), op1.data(), op2.data(), op1.size(), 1));
  EXPECT_ANY_THROW(EltwiseAddMod(op1.data(), big_input.data(), op2.data(),
                                 op1.size(), modulus));
  EXPECT_ANY_THROW(EltwiseAddMod(op1.data(), op1.data(), big_input.data(),
                                 op1.size(), modulus));
}

TEST(EltwiseAddMod, vector_scalar_bad_input) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t op2{1};
  std::vector<uint64_t> big_input{11, 12, 13, 14, 15, 16, 17, 18};
  uint64_t modulus = 10;

  EXPECT_ANY_THROW(
      EltwiseAddMod(nullptr, op1.data(), op2, op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseAddMod(op1.data(), nullptr, op2, op1.size(), modulus));
  EXPECT_ANY_THROW(
      EltwiseAddMod(op1.data(), op1.data(), modulus, op1.size(), modulus));
  EXPECT_ANY_THROW(EltwiseAddMod(op1.data(), op1.data(), op2, 0, modulus));
  EXPECT_ANY_THROW(EltwiseAddMod(op1.data(), op1.data(), op2, op1.size(), 1));
  EXPECT_ANY_THROW(
      EltwiseAddMod(op1.data(), big_input.data(), op2, op1.size(), modulus));
}
#endif

TEST(EltwiseAddMod, vector_vector_native_small) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 9, 4, 4, 6};
  std::vector<uint64_t> exp_out{2, 5, 8, 1, 4, 0, 1, 4};
  uint64_t modulus = 10;

  EltwiseAddModNative(op1.data(), op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseAddMod, vector_scalar_native_small) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t op2{3};
  std::vector<uint64_t> exp_out{4, 5, 6, 7, 8, 9, 0, 1};
  uint64_t modulus = 10;

  EltwiseAddModNative(op1.data(), op1.data(), op2, op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseAddMod, vector_vector_native_big) {
  uint64_t modulus = GeneratePrimes(1, 60, true, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, modulus - 1, modulus - 2, modulus - 2,
                            modulus - 3, modulus - 3, modulus - 4, modulus - 4};
  std::vector<uint64_t> op2{modulus - 1, modulus - 2, modulus - 3, modulus - 4,
                            modulus - 5, modulus - 6, modulus - 7, modulus - 8};
  std::vector<uint64_t> exp_out{modulus - 2,  modulus - 3, modulus - 5,
                                modulus - 6,  modulus - 8, modulus - 9,
                                modulus - 11, modulus - 12};

  EltwiseAddModNative(op1.data(), op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseAddMod, vector_scalar_native_big) {
  uint64_t modulus = GeneratePrimes(1, 60, true, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, modulus - 1, modulus - 2, modulus - 2,
                            modulus - 3, modulus - 3, modulus - 4, modulus - 4};
  uint64_t op2{modulus - 1};
  std::vector<uint64_t> exp_out{modulus - 2, modulus - 2, modulus - 3,
                                modulus - 3, modulus - 4, modulus - 4,
                                modulus - 5, modulus - 5};

  EltwiseAddModNative(op1.data(), op1.data(), op2, op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

}  // namespace hexl
}  // namespace intel
