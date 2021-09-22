// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-add-mod-avx512.hpp"
#include "eltwise/eltwise-add-mod-internal.hpp"
#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util-avx512.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseAddMod, vector_vector_avx512_small) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 9, 2, 4, 6};
  std::vector<uint64_t> exp_out{2, 5, 8, 1, 4, 8, 1, 4};
  uint64_t modulus = 10;
  EltwiseAddModAVX512(op1.data(), op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseAddMod, vector_scalar_avx512_small) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t op2{3};
  std::vector<uint64_t> exp_out{4, 5, 6, 7, 8, 9, 0, 1};
  uint64_t modulus = 10;
  EltwiseAddModAVX512(op1.data(), op1.data(), op2, op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseAddMod, vector_vector_avx512_big) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  uint64_t modulus = GeneratePrimes(1, 60, true, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, modulus - 1, modulus - 2, modulus - 2,
                            modulus - 3, modulus - 3, modulus - 4, modulus - 4};
  std::vector<uint64_t> op2{modulus - 1, modulus - 2, modulus - 3, modulus - 4,
                            modulus - 5, modulus - 6, modulus - 7, modulus - 8};
  std::vector<uint64_t> exp_out{modulus - 2,  modulus - 3, modulus - 5,
                                modulus - 6,  modulus - 8, modulus - 9,
                                modulus - 11, modulus - 12};

  EltwiseAddModAVX512(op1.data(), op1.data(), op2.data(), op1.size(), modulus);

  CheckEqual(op1, exp_out);
}

TEST(EltwiseAddMod, vector_scalar_avx512_big) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  uint64_t modulus = GeneratePrimes(1, 60, true, 1024)[0];

  std::vector<uint64_t> op1{modulus - 1, modulus - 1, modulus - 2, modulus - 2,
                            modulus - 3, modulus - 3, modulus - 4, modulus - 4};
  uint64_t op2{modulus - 1};
  std::vector<uint64_t> exp_out{modulus - 2, modulus - 2, modulus - 3,
                                modulus - 3, modulus - 4, modulus - 4,
                                modulus - 5, modulus - 5};

  EltwiseAddModAVX512(op1.data(), op1.data(), op2, op1.size(), modulus);

  CheckEqual(op1, exp_out);
}
#endif

// Checks AVX512 and native eltwise add implementations match
#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseAddMod, vector_vector_avx512_native_match) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  size_t length = 173;

  for (size_t bits = 1; bits <= 62; ++bits) {
    uint64_t modulus = 1ULL << bits;

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 100;
#endif

    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformRandomValues(length, 0, modulus);
      auto op2 = GenerateInsecureUniformRandomValues(length, 0, modulus);
      op1[0] = modulus - 1;
      op2[0] = modulus - 1;

      auto op1a = op1;

      EltwiseAddModNative(op1.data(), op1.data(), op2.data(), op1.size(),
                          modulus);
      EltwiseAddModAVX512(op1a.data(), op1a.data(), op2.data(), op1.size(),
                          modulus);

      ASSERT_EQ(op1, op1a);
      ASSERT_EQ(op1[0], modulus - 2);
      ASSERT_EQ(op1a[0], modulus - 2);
    }
  }
}

TEST(EltwiseAddMod, vector_scalar_avx512_native_match) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  size_t length = 173;

  for (size_t bits = 1; bits <= 62; ++bits) {
    uint64_t modulus = 1ULL << bits;

#ifdef HEXL_DEBUG
    size_t num_trials = 10;
#else
    size_t num_trials = 10000;
#endif

    for (size_t trial = 0; trial < num_trials; ++trial) {
      auto op1 = GenerateInsecureUniformRandomValues(length, 0, modulus);
      uint64_t op2 = GenerateInsecureUniformRandomValue(0, modulus);

      auto op1a = op1;

      EltwiseAddModNative(op1.data(), op1.data(), op2, op1.size(), modulus);
      EltwiseAddModAVX512(op1a.data(), op1a.data(), op2, op1.size(), modulus);

      ASSERT_EQ(op1, op1a);
    }
  }
}
#endif

}  // namespace hexl
}  // namespace intel
