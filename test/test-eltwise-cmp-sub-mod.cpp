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
#include "test-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(EltwiseCmpSubMod, null) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t modulus{10};

  EXPECT_ANY_THROW(EltwiseCmpSubMod(nullptr, op1.data(), op1.size(), modulus,
                                    CMPINT::EQ, 1, 1));
  EXPECT_ANY_THROW(EltwiseCmpSubMod(op1.data(), nullptr, op1.size(), modulus,
                                    CMPINT::EQ, 1, 1));
  EXPECT_ANY_THROW(
      EltwiseCmpSubMod(op1.data(), op1.data(), 0, modulus, CMPINT::EQ, 1, 1));
  EXPECT_ANY_THROW(EltwiseCmpSubMod(op1.data(), op1.data(), op1.size(), modulus,
                                    CMPINT::EQ, 1, 0));
  EXPECT_ANY_THROW(EltwiseCmpSubMod(op1.data(), op1.data(), op1.size(), modulus,
                                    CMPINT::EQ, 1, 0));
}
#endif

// Parameters = (input, modulus, cmp, bound, diff, expected_output)
class EltwiseCmpSubModTest
    : public ::testing::TestWithParam<
          std::tuple<std::vector<uint64_t>, uint64_t, CMPINT, uint64_t,
                     uint64_t, std::vector<uint64_t>>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test Native implementation
TEST_P(EltwiseCmpSubModTest, Native) {
  std::vector<uint64_t> input = std::get<0>(GetParam());
  uint64_t modulus = std::get<1>(GetParam());
  CMPINT cmp = std::get<2>(GetParam());
  uint64_t bound = std::get<3>(GetParam());
  uint64_t diff = std::get<4>(GetParam());
  std::vector<uint64_t> exp_output = std::get<5>(GetParam());

  EltwiseCmpSubModNative(input.data(), input.data(), input.size(), modulus, cmp,
                         bound, diff);

  CheckEqual(input, exp_output);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseCmpSubModTest, EltwiseCmpSubModTest,
    ::testing::Values(
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, 10,
                        CMPINT::EQ, 4, 5,
                        std::vector<uint64_t>{1, 2, 3, 9, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, 10,
                        CMPINT::LT, 4, 5,
                        std::vector<uint64_t>{6, 7, 8, 4, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, 10,
                        CMPINT::LE, 4, 5,
                        std::vector<uint64_t>{6, 7, 8, 9, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, 10,
                        CMPINT::FALSE, 4, 5,
                        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, 10,
                        CMPINT::NE, 4, 5,
                        std::vector<uint64_t>{6, 7, 8, 4, 0, 1, 2}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, 10,
                        CMPINT::NLT, 4, 5,
                        std::vector<uint64_t>{1, 2, 3, 9, 0, 1, 2}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, 10,
                        CMPINT::NLE, 4, 5,
                        std::vector<uint64_t>{1, 2, 3, 4, 0, 1, 2}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, 10,
                        CMPINT::TRUE, 4, 5,
                        std::vector<uint64_t>{6, 7, 8, 9, 0, 1, 2})));

// Checks AVX512 and native implementations match
#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseCmpSubMod, AVX512) {
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
