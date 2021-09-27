// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "ntt/ntt-internal.hpp"
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

struct ModulusInputModData {
  explicit ModulusInputModData(std::tuple<uint64_t, bool, uint64_t> param) {
    modulus_bits = std::get<0>(param);
    prefer_small_modulus = std::get<1>(param);
    input_mod_factor = std::get<2>(param);
  }

  uint64_t modulus_bits;
  bool prefer_small_modulus;
  uint64_t input_mod_factor;
};

class ModulusInputModFactor
    : public ::testing::TestWithParam<std::tuple<uint64_t, bool, uint64_t>> {
 public:
  struct PrintToStringParamName {
    template <class ParamType>
    std::string operator()(
        const testing::TestParamInfo<ParamType>& info) const {
      ModulusInputModData modulus_data(
          static_cast<std::tuple<uint64_t, bool, uint64_t>>(info.param));

      std::stringstream ss;
      ss << "q" << std::to_string(modulus_data.modulus_bits)
         << "bits_SmallPrimes"
         << std::to_string(modulus_data.prefer_small_modulus)
         << "_InputModFactor" << std::to_string(modulus_data.input_mod_factor);

      return ss.str();
    }
  };

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_P(ModulusInputModFactor, NativeRandom) {
  ModulusInputModData modulus_data(GetParam());

  uint64_t modulus = GeneratePrimes(1, modulus_data.modulus_bits,
                                    modulus_data.prefer_small_modulus)[0];
  uint64_t length = 1024;

  uint64_t data_bound = modulus_data.input_mod_factor;
  auto input_1 = GenerateInsecureUniformRandomValues(length, 0, data_bound);
  auto input_2 = GenerateInsecureUniformRandomValues(length, 0, data_bound);
  std::vector<uint64_t> output(length, 0);

  std::vector<uint64_t> expected(length, 0);
  for (size_t i = 0; i < length; ++i) {
    expected[i] = MultiplyMod(input_1[i], input_2[i], modulus);
  }

  switch (modulus_data.input_mod_factor) {
    case 1: {
      EltwiseMultModNative<1>(output.data(), input_1.data(), input_2.data(),
                              length, modulus);
      break;
    }
    case 2: {
      EltwiseMultModNative<2>(output.data(), input_1.data(), input_2.data(),
                              length, modulus);
      break;
    }
    case 4: {
      EltwiseMultModNative<4>(output.data(), input_1.data(), input_2.data(),
                              length, modulus);
      break;
    }
  }
  ASSERT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseMultMod, ModulusInputModFactor,
    ::testing::Combine(::testing::Range(uint64_t{30}, uint64_t{61}),
                       ::testing::ValuesIn(std::vector<bool>{false, true}),
                       ::testing::ValuesIn(std::vector<uint64_t>{1, 2, 4})),
    ModulusInputModFactor::PrintToStringParamName());

}  // namespace hexl
}  // namespace intel
