// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-reduce-mod-avx512.hpp"
#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test/test-util-avx512.hpp"
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

TEST(EltwiseReduceModMontInOut, avx512_64_mod_1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  uint64_t modulus = 67280421310725ULL;
  std::vector<uint64_t> input_a{0,
                                67280421310000,
                                25040294381203,
                                340231313,
                                769231483400,
                                90032324,
                                120042353,
                                1530};
  std::vector<uint64_t> output{0, 0, 0, 0, 0, 0, 0, 0};

  int r = 46;  // R^2 mod N = 42006526039321
  uint64_t R_reduced = ReduceMod<2>(1ULL << r, modulus);
  const uint64_t R_square_mod_q = MultiplyMod(R_reduced, R_reduced, modulus);
  uint64_t inv_mod = HenselLemma2adicRoot(r, modulus);

  EltwiseMontgomeryFormInAVX512<64, 46>(output.data(), input_a.data(),
                                        R_square_mod_q, input_a.size(), modulus,
                                        inv_mod);
  EltwiseMontgomeryFormOutAVX512<64, 46>(output.data(), output.data(),
                                         input_a.size(), modulus, inv_mod);
  CheckEqual(input_a, output);
}

#ifdef HEXL_HAS_AVX512IFMA

TEST(EltwiseReduceMod, avx512_52_mod_1) {
  if (!has_avx512ifma) {
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

TEST(EltwiseReduceMod, avx512_52_Big_mod_1) {
  if (!has_avx512ifma) {
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

TEST(EltwiseReduceModMontInOut, avx512_52_mod_1) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }

  uint64_t modulus = 67280421310725ULL;
  std::vector<uint64_t> input_a{0,
                                67280421310000,
                                25040294381203,
                                340231313,
                                769231483400,
                                90032324,
                                120042353,
                                1530};
  std::vector<uint64_t> output{0, 0, 0, 0, 0, 0, 0, 0};

  int r = 46;  // R^2 mod N = 42006526039321
  uint64_t R_reduced = ReduceMod<2>(1ULL << r, modulus);
  const uint64_t R_square_mod_q = MultiplyMod(R_reduced, R_reduced, modulus);
  uint64_t inv_mod = HenselLemma2adicRoot(r, modulus);

  EltwiseMontgomeryFormInAVX512<52, 46>(output.data(), input_a.data(),
                                        R_square_mod_q, input_a.size(), modulus,
                                        inv_mod);
  EltwiseMontgomeryFormOutAVX512<52, 46>(output.data(), output.data(),
                                         input_a.size(), modulus, inv_mod);
  CheckEqual(input_a, output);
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
      auto op1 = GenerateInsecureUniformIntRandomValues(length, 0, 1ULL << 63);
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
      auto op1 = GenerateInsecureUniformIntRandomValues(length, 0, 4 * modulus);
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
      auto op1 = GenerateInsecureUniformIntRandomValues(length, 0, 4 * modulus);
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
      auto op1 = GenerateInsecureUniformIntRandomValues(length, 0, 2 * modulus);
      auto op2 = op1;
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus, 2,
                             1);
      EltwiseReduceModAVX512(result2.data(), op2.data(), op1.size(), modulus, 2,
                             1);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

#ifdef HEXL_HAS_AVX512IFMA
// Checks AVX512 and native EltwiseReduceMod implementations match with randomly
// generated inputs
TEST(EltwiseReduceMod, AVX512_52_Big_0_1) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }

  size_t length = 8;

  for (size_t bits = 45; bits <= 51; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];
#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 1;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformIntRandomValues(length, 0, 1ULL << 63);
      auto op2 = op1;

      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus,
                             modulus, 1);
      EltwiseReduceModAVX512<52>(result2.data(), op2.data(), op1.size(),
                                 modulus, modulus, 1);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

TEST(EltwiseReduceMod, AVX512_52_Big_4_1) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }

  size_t length = 8;

  for (size_t bits = 45; bits <= 52; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];
#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 1;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformIntRandomValues(length, 0, 4 * modulus);
      auto op2 = op1;
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus, 4,
                             1);
      EltwiseReduceModAVX512<52>(result2.data(), op2.data(), op1.size(),
                                 modulus, 4, 1);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

TEST(EltwiseReduceMod, AVX512_52_Big_4_2) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }

  size_t length = 8;

  for (size_t bits = 45; bits <= 52; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];
#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 1;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformIntRandomValues(length, 0, 4 * modulus);
      auto op2 = op1;
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus, 4,
                             2);
      EltwiseReduceModAVX512<52>(result2.data(), op2.data(), op1.size(),
                                 modulus, 4, 2);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

TEST(EltwiseReduceMod, AVX512_52_Big_2_1) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }

  size_t length = 8;

  for (size_t bits = 45; bits <= 52; ++bits) {
    uint64_t modulus = GeneratePrimes(1, bits, true, length)[0];
#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 1;
#endif
    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformIntRandomValues(length, 0, 2 * modulus);
      auto op2 = op1;
      std::vector<uint64_t> result1(length, 0);
      std::vector<uint64_t> result2(length, 0);

      EltwiseReduceModNative(result1.data(), op1.data(), op1.size(), modulus, 2,
                             1);
      EltwiseReduceModAVX512<52>(result2.data(), op2.data(), op1.size(),
                                 modulus, 2, 1);

      ASSERT_EQ(result1, result2);
      ASSERT_EQ(result1, result2);
    }
  }
}

#endif

#endif

}  // namespace hexl
}  // namespace intel
