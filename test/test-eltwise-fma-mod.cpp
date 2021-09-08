// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

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

}  // namespace hexl
}  // namespace intel
