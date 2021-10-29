// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

TEST(EltwiseReduceMod, 2_2) {
  std::vector<uint64_t> op{0, 450, 735, 900, 1350, 1459};
  std::vector<uint64_t> exp_out{0, 450, 735, 900, 1350, 1459};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0};

  const uint64_t modulus = 750;
  const uint64_t input_mod_factor = 2;
  const uint64_t output_mod_factor = 2;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, 4_1) {
  std::vector<uint64_t> op{2, 4, 1600, 2500};
  std::vector<uint64_t> exp_out{2, 4, 100, 250};
  std::vector<uint64_t> result{0, 0, 0, 0};

  const uint64_t modulus = 750;
  const uint64_t input_mod_factor = 4;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, 0_1) {
  std::vector<uint64_t> op{2, 4, 1600, 2500};
  std::vector<uint64_t> exp_out{2, 4, 100, 250};
  std::vector<uint64_t> result{0, 0, 0, 0};

  const uint64_t modulus = 750;
  const uint64_t input_mod_factor = modulus;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, 2_1) {
  std::vector<uint64_t> op{0, 450, 735, 900, 1350, 1459};
  std::vector<uint64_t> exp_out{0, 450, 5, 170, 620, 729};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0};

  const uint64_t modulus = 730;
  const uint64_t input_mod_factor = 2;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, 4_2) {
  std::vector<uint64_t> op{1, 730, 1000, 1460, 2100, 2919};
  std::vector<uint64_t> exp_out{1, 730, 1000, 0, 640, 1459};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0};

  const uint64_t modulus = 730;
  const uint64_t input_mod_factor = 4;
  const uint64_t output_mod_factor = 2;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

// First parameter is the number of bits in the modulus
// Second parameter is whether or not to prefer small moduli
class EltwiseReduceModTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, bool>> {
 protected:
  void SetUp() override {
    m_modulus_bits = std::get<0>(GetParam());
    m_prefer_small_primes = std::get<1>(GetParam());
    m_modulus = GeneratePrimes(1, m_modulus_bits, m_prefer_small_primes)[0];
  }

  void TearDown() override {}

 public:
  uint64_t m_N{1024 + 7};  // m_N % 8 = 7 to test AVX512 boundary case
  uint64_t m_modulus_bits;
  bool m_prefer_small_primes;
  uint64_t m_modulus;
};

// Test public API matches Native implementation on random values
TEST_P(EltwiseReduceModTest, Random) {
  uint64_t upper_bound =
      m_modulus < (1ULL << 32) ? m_modulus * m_modulus : 1ULL << 63;

  auto input = GenerateInsecureUniformRandomValues(m_N, 0, upper_bound);
  std::vector<uint64_t> result_native(m_N, 0);
  std::vector<uint64_t> result_public_api(m_N, 0);

  EltwiseReduceModNative(result_native.data(), input.data(), m_N, m_modulus,
                         m_modulus, 1);
  EltwiseReduceMod(result_public_api.data(), input.data(), m_N, m_modulus,
                   m_modulus, 1);
  AssertEqual(result_native, result_public_api);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseReduceMod, EltwiseReduceModTest,
    ::testing::Combine(::testing::ValuesIn(AlignedVector64<uint64_t>{
                           20, 25, 30, 31, 32, 33, 35, 40, 48, 49, 50, 51, 52,
                           55, 58, 59, 60}),
                       ::testing::ValuesIn(std::vector<bool>{false, true})));

}  // namespace hexl
}  // namespace intel
