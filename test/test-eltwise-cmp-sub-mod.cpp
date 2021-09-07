// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

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

}  // namespace hexl
}  // namespace intel
