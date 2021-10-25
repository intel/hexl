// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-reduce-mod-avx512.hpp"
#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util-avx512.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseReduceMod, avx512_64_mod_1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> op{0, 111, 250, 340, 769, 900, 1200, 1530};
  std::vector<uint64_t> exp_out{0, 111, 250, 340, 0, 131, 431, 761};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 769;
  const uint64_t input_mod_factor = modulus;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceModAVX512<64>(result.data(), op.data(), op.size(), modulus,
                             input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

#ifdef HEXL_HAS_AVX512IFMA
TEST(EltwiseReduceMod, avx512_52_mod_1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> op{0, 111, 250, 340, 769, 900, 1200, 1530};
  std::vector<uint64_t> exp_out{0, 111, 250, 340, 0, 131, 431, 761};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 769;
  const uint64_t input_mod_factor = modulus;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceModAVX512<52>(result.data(), op.data(), op.size(), modulus,
                             input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, avx512Big_mod_1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> op{914704788761805005, 224925333812073588,
                           592788284123677125, 142439467624940029,
                           146023272535470246, 979015887843024185,
                           496780369302017539, 1073741441};
  std::vector<uint64_t> exp_out{802487803, 754009873, 962097738, 36142730,
                                687617508, 519876583, 630345322, 0};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};

  uint64_t modulus = 1073741441;
  const uint64_t input_mod_factor = modulus;
  const uint64_t output_mod_factor = 1;

  EltwiseReduceModAVX512<52>(result.data(), op.data(), op.size(), modulus,
                             input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}
#endif

TEST(EltwiseReduceMod, avx512_2_1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

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
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

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
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

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
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  size_t length = 1024;

  for (size_t bits = 50; bits <= 62; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformRandomValues(length, 0, modulus);
      auto op2 = op1;

      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus,
                             modulus, 1);
      EltwiseReduceModAVX512(result2.data(), op2.data(), op1.size(), modulus,
                             modulus, 1);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

TEST(EltwiseReduceMod, AVX512Big_4_1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  size_t length = 1024;

  for (size_t bits = 50; bits <= 62; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformRandomValues(length, 0, 4 * modulus);
      auto op2 = op1;
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

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
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  size_t length = 1024;

  for (size_t bits = 50; bits <= 62; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformRandomValues(length, 0, 4 * modulus);
      auto op2 = op1;
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

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
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  size_t length = 1024;

  for (size_t bits = 50; bits <= 62; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformRandomValues(length, 0, 2 * modulus);
      auto op2 = op1;
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

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
