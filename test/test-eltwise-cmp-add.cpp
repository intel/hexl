// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "hexl/eltwise/eltwise-cmp-add.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(EltwiseCmpAdd, null) {
  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};

  EXPECT_ANY_THROW(
      EltwiseCmpAdd(nullptr, op1.data(), op1.size(), CMPINT::EQ, 1, 1));
  EXPECT_ANY_THROW(
      EltwiseCmpAdd(op1.data(), nullptr, op1.size(), CMPINT::EQ, 1, 1));
  EXPECT_ANY_THROW(EltwiseCmpAdd(op1.data(), op1.data(), 0, CMPINT::EQ, 1, 1));
  EXPECT_ANY_THROW(
      EltwiseCmpAdd(op1.data(), op1.data(), op1.size(), CMPINT::EQ, 1, 0));
}
#endif

// Parameters = (input, cmp, bound, diff, expected_output)
class EltwiseCmpAddTest
    : public ::testing::TestWithParam<
          std::tuple<std::vector<uint64_t>, CMPINT, uint64_t, uint64_t,
                     std::vector<uint64_t>>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test Native implementation
TEST_P(EltwiseCmpAddTest, Native) {
  std::vector<uint64_t> input = std::get<0>(GetParam());
  CMPINT cmp = std::get<1>(GetParam());
  uint64_t bound = std::get<2>(GetParam());
  uint64_t diff = std::get<3>(GetParam());
  std::vector<uint64_t> exp_output = std::get<4>(GetParam());

  EltwiseCmpAddNative(input.data(), input.data(), input.size(), cmp, bound,
                      diff);

  CheckEqual(input, exp_output);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseCmpAddTest, EltwiseCmpAddTest,
    ::testing::Values(
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::EQ,
                        4, 5, std::vector<uint64_t>{1, 2, 3, 9, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::LT,
                        4, 5, std::vector<uint64_t>{6, 7, 8, 4, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::LE,
                        4, 5, std::vector<uint64_t>{6, 7, 8, 9, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7},
                        CMPINT::FALSE, 4, 5,
                        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NE,
                        4, 5, std::vector<uint64_t>{6, 7, 8, 4, 10, 11, 12}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NLT,
                        4, 5, std::vector<uint64_t>{1, 2, 3, 9, 10, 11, 12}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7}, CMPINT::NLE,
                        4, 5, std::vector<uint64_t>{1, 2, 3, 4, 10, 11, 12}),
        std::make_tuple(std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7},
                        CMPINT::TRUE, 4, 5,
                        std::vector<uint64_t>{6, 7, 8, 9, 10, 11, 12})));

}  // namespace hexl
}  // namespace intel
