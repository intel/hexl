// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "eltwise/eltwise-dot-mod-internal.hpp"
#include "hexl/eltwise/eltwise-dot-mod.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

// Parameters = (input, modulus, cmp, bound, diff, expected_output)
class EltwiseDotModTest
    : public ::testing::TestWithParam<std::tuple<
          std::vector<uint64_t>, std::vector<uint64_t>, std::vector<uint64_t>,
          std::vector<uint64_t>, std::vector<uint64_t>, uint64_t, uint64_t>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test Native implementation
TEST_P(EltwiseDotModTest, Native) {
  std::vector<uint64_t> expected = std::get<0>(GetParam());
  std::vector<uint64_t> operand1 = std::get<1>(GetParam());
  std::vector<uint64_t> operand2 = std::get<2>(GetParam());
  std::vector<uint64_t> operand3 = std::get<3>(GetParam());
  std::vector<uint64_t> operand4 = std::get<4>(GetParam());
  uint64_t n = std::get<5>(GetParam());
  uint64_t modulus = std::get<6>(GetParam());

  EltwiseDotModNative(operand1.data(), operand1.data(), operand2.data(),
                      operand3.data(), operand4.data(), n, modulus);

  CheckEqual(expected, operand1);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseDotModTest, EltwiseDotModTest,
    ::testing::Values(std::make_tuple(
        std::vector<uint64_t>{34, 88, 46, 8, 74, 44, 18, 96},
        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8},
        std::vector<uint64_t>{9, 10, 11, 12, 13, 14, 15, 16},
        std::vector<uint64_t>{17, 18, 19, 20, 21, 22, 23, 24},
        std::vector<uint64_t>{25, 26, 27, 28, 29, 30, 31, 32}, 8, 100)));

}  // namespace hexl
}  // namespace intel
