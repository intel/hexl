// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "eltwise/eltwise-reduce-mod-avx512.hpp"
#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"

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
  const uint64_t input_mod_factor = 0;
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

#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseReduceMod, avx512_0_1) {
  std::vector<uint64_t> op{0, 111, 250, 340, 769, 900, 1200, 1530};
  std::vector<uint64_t> exp_out{0, 111, 250, 340, 0, 131, 431, 761};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 769;
  const uint64_t input_mod_factor = 0;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceModAVX512(result.data(), op.data(), op.size(), modulus,
                         input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, avx512_2_1) {
  std::vector<uint64_t> op{0, 54, 100, 135, 201, 18, 148, 168, 201};
  std::vector<uint64_t> exp_out{0, 54, 100, 34, 100, 18, 47, 67, 100};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 101;
  const uint64_t input_mod_factor = 2;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceModAVX512(result.data(), op.data(), op.size(), modulus,
                         input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, avx512_4_1) {
  std::vector<uint64_t> op{0, 54, 100, 135, 201, 220, 350, 370, 403};
  std::vector<uint64_t> exp_out{0, 54, 100, 34, 100, 18, 47, 67, 100};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 101;
  const uint64_t input_mod_factor = 4;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceModAVX512(result.data(), op.data(), op.size(), modulus,
                         input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, avx512_4_2) {
  std::vector<uint64_t> op{0, 54, 100, 135, 201, 220, 350, 370, 403};
  std::vector<uint64_t> exp_out{0, 54, 100, 135, 201, 18, 148, 168, 201};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 101;
  const uint64_t input_mod_factor = 4;
  const uint64_t output_mod_factor = 2;
  EltwiseReduceModAVX512(result.data(), op.data(), op.size(), modulus,
                         input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

// Checks AVX512 and native EltwiseReduceMod implementations match with randomly
// generated inputs
TEST(EltwiseReduceMod, AVX512Big_0_1) {
  std::random_device rd;
  std::mt19937 gen(rd());

  size_t length = 1024;

  for (size_t bits = 50; bits <= 62; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, length)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, modulus - 1);

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      std::vector<uint64_t> op1(length, 0);
      for (size_t i = 0; i < length; ++i) {
        op1[i] = distrib(gen);
      }
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);
      auto op2 = op1;

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus, 0,
                             1);
      EltwiseReduceModAVX512(result2.data(), op2.data(), op1.size(), modulus, 0,
                             1);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

TEST(EltwiseReduceMod, AVX512Big_4_1) {
  std::random_device rd;
  std::mt19937 gen(rd());

  size_t length = 1024;

  for (size_t bits = 50; bits <= 62; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, length)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, (4 * modulus) - 1);

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      std::vector<uint64_t> op1(length, 0);
      for (size_t i = 0; i < length; ++i) {
        op1[i] = distrib(gen);
      }
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);
      auto op2 = op1;

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus, 4,
                             1);
      EltwiseReduceModAVX512(result2.data(), op2.data(), op1.size(), modulus, 4,
                             1);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

TEST(EltwiseReduceMod, AVX512Big_4_2) {
  std::random_device rd;
  std::mt19937 gen(rd());

  size_t length = 1024;

  for (size_t bits = 50; bits <= 62; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, length)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, (4 * modulus) - 1);

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      std::vector<uint64_t> op1(length, 0);
      for (size_t i = 0; i < length; ++i) {
        op1[i] = distrib(gen);
      }
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);
      auto op2 = op1;

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus, 4,
                             2);
      EltwiseReduceModAVX512(result2.data(), op2.data(), op1.size(), modulus, 4,
                             2);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

TEST(EltwiseReduceMod, AVX512Big_2_1) {
  std::random_device rd;
  std::mt19937 gen(rd());

  size_t length = 1024;

  for (size_t bits = 50; bits <= 62; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, length)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, (2 * modulus) - 1);

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      std::vector<uint64_t> op1(length, 0);
      for (size_t i = 0; i < length; ++i) {
        op1[i] = distrib(gen);
      }
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);
      auto op2 = op1;

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus, 4,
                             1);
      EltwiseReduceModAVX512(result2.data(), op2.data(), op1.size(), modulus, 4,
                             1);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}
#endif

}  // namespace hexl
}  // namespace intel
