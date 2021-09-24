// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util-avx512.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseMultMod, avx512_small) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  std::vector<uint64_t> op1{1, 2, 3, 1, 1, 1, 0, 1, 0};
  std::vector<uint64_t> op2{1, 1, 1, 1, 2, 3, 1, 0, 0};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{1, 2, 3, 1, 2, 3, 0, 0, 0};

  uint64_t modulus = 769;
  EltwiseMultModAVX512Float<1>(result.data(), op1.data(), op2.data(),
                               op1.size(), modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, avx512_int2) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  uint64_t modulus = GeneratePrimes(1, 60, true, 1024)[0];

  std::vector<uint64_t> op1{modulus - 3, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> op2{modulus - 4, 1, 1, 1, 1, 1, 1, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{12, 1, 1, 1, 1, 1, 1, 1};

  EltwiseMultModAVX512DQInt<2>(result.data(), op1.data(), op2.data(),
                               op1.size(), modulus);
  CheckEqual(result, exp_out);
}

#endif

#ifdef HEXL_HAS_AVX512DQ
TEST(EltwiseMultMod, Big) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  uint64_t modulus = 1125891450734593;

  std::vector<uint64_t> op1{706712574074152, 943467560561867, 1115920708919443,
                            515713505356094, 525633777116309, 910766532971356,
                            757086506562426, 799841520990167, 1};
  std::vector<uint64_t> op2{515910833966633, 96924929169117,   537587376997453,
                            41829060600750,  205864998008014,  463185427411646,
                            965818279134294, 1075778049568657, 1};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint64_t> exp_out{
      231838787758587, 618753612121218, 1116345967490421,
      409735411065439, 25680427818594,  950138933882289,
      554128714280822, 1465109636753,   1};

  EltwiseMultModAVX512DQInt<4>(result.data(), op1.data(), op2.data(),
                               op1.size(), modulus);

  CheckEqual(result, exp_out);
}

TEST(EltwiseMultMod, AVX512FloatInPlaceNoInputReduceMod) {
  uint64_t modulus = 281474976546817;

  std::vector<uint64_t> data_native(8, 998771110802331);
  auto data_avx = data_native;

  EltwiseMultModAVX512Float<4>(data_avx.data(), data_avx.data(),
                               data_avx.data(), data_avx.size(), modulus);

  EltwiseMultModNative<4>(data_native.data(), data_native.data(),
                          data_native.data(), data_avx.size(), modulus);

  CheckEqual(data_native, std::vector<uint64_t>(8, 273497826869315));
  CheckEqual(data_avx, std::vector<uint64_t>(8, 273497826869315));
  CheckEqual(data_avx, data_native);
}

TEST(EltwiseMultMod, avx512dqint_small) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  uint64_t input_mod_factor = 1;
  uint64_t modulus = (1ULL << 53) + 7;

  for (size_t length = 1024; length <= 32768; length *= 2) {
    auto op1 = GenerateInsecureUniformRandomValues(length, 0,
                                                   input_mod_factor * modulus);
    auto op2 = GenerateInsecureUniformRandomValues(length, 0,
                                                   input_mod_factor * modulus);

    std::vector<uint64_t> out_avx(length, 0);
    std::vector<uint64_t> out_native(length, 0);

    EltwiseMultModAVX512DQInt<1>(out_avx.data(), op1.data(), op2.data(),
                                 op1.size(), modulus);

    EltwiseMultModNative<1>(out_native.data(), op1.data(), op2.data(),
                            op1.size(), modulus);

    CheckEqual(out_avx, out_native);
  }
}

// Checks AVX512 and native eltwise mult out-of-place implementations match
TEST(EltwiseMultMod, avx512dqint_big) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  for (size_t length = 1024; length <= 32768; length *= 2) {
    std::vector<uint64_t> op1(length, 0);
    std::vector<uint64_t> op2(length, 0);
    std::vector<uint64_t> rs1(length, 0);
    std::vector<uint64_t> rs2(length, 0);
    std::vector<uint64_t> rs3(length, 0);
    std::vector<uint64_t> rs4(length, 0);

    for (size_t input_mod_factor = 1; input_mod_factor <= 4;
         input_mod_factor *= 2) {
      for (size_t bits = 40; bits <= 60; ++bits) {
        uint64_t modulus = (1ULL << bits) + 7;
        uint64_t data_upper_bound = input_mod_factor * modulus;
        bool use_avx512_float = (data_upper_bound < MaximumValue(50));

        size_t num_trials = 1;
        for (size_t trial = 0; trial < num_trials; ++trial) {
          auto op1 =
              GenerateInsecureUniformRandomValues(length, 0, data_upper_bound);
          auto op2 =
              GenerateInsecureUniformRandomValues(length, 0, data_upper_bound);

          op1[0] = data_upper_bound - 1;
          op2[0] = data_upper_bound - 1;

          switch (input_mod_factor) {
            case 1:
              EltwiseMultModNative<1>(rs1.data(), op1.data(), op2.data(),
                                      op1.size(), modulus);
              if (use_avx512_float) {
                EltwiseMultModAVX512Float<1>(rs2.data(), op1.data(), op2.data(),
                                             op1.size(), modulus);
              } else {
                EltwiseMultModAVX512DQInt<1>(rs3.data(), op1.data(), op2.data(),
                                             op1.size(), modulus);
              }
              break;
            case 2:
              EltwiseMultModNative<2>(rs1.data(), op1.data(), op2.data(),
                                      op1.size(), modulus);
              if (use_avx512_float) {
                EltwiseMultModAVX512Float<2>(rs2.data(), op1.data(), op2.data(),
                                             op1.size(), modulus);
              } else {
                EltwiseMultModAVX512DQInt<2>(rs3.data(), op1.data(), op2.data(),
                                             op1.size(), modulus);
              }
              break;
            case 4:
              EltwiseMultModNative<4>(rs1.data(), op1.data(), op2.data(),
                                      op1.size(), modulus);
              if (use_avx512_float) {
                EltwiseMultModAVX512Float<4>(rs2.data(), op1.data(), op2.data(),
                                             op1.size(), modulus);
              } else {
                EltwiseMultModAVX512DQInt<4>(rs3.data(), op1.data(), op2.data(),
                                             op1.size(), modulus);
              }
              break;
          }
          EltwiseMultMod(rs4.data(), op1.data(), op2.data(), op1.size(),
                         modulus, input_mod_factor);

          ASSERT_EQ(rs4, rs1);

          ASSERT_EQ(rs1[0], 1);
          if (use_avx512_float) {
            ASSERT_EQ(rs1, rs2);
            ASSERT_EQ(rs2[0], 1);
          } else {
            ASSERT_EQ(rs1, rs3);
            ASSERT_EQ(rs3[0], 1);
          }
        }
      }
    }
  }
}
#endif

#ifdef HEXL_HAS_AVX512IFMA
TEST(EltwiseMultMod, avx512ifma_big) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }

  for (size_t length = 8; length <= 8; length *= 2) {
    std::vector<uint64_t> op1(length, 0);
    std::vector<uint64_t> op2(length, 0);
    std::vector<uint64_t> result_native(length, 0);
    std::vector<uint64_t> result_ifma(length, 0);

    for (size_t input_mod_factor = 1; input_mod_factor <= 4;
         input_mod_factor *= 2) {
      for (size_t bits = 40; bits <= 50; ++bits) {
        uint64_t modulus = (1ULL << bits) + 7;
        uint64_t data_upper_bound = input_mod_factor * modulus;
        if (data_upper_bound > MaximumValue(50)) {
          continue;
        }

        HEXL_VLOG(2,
                  "bits " << bits << " input_mod_factor " << input_mod_factor);

#ifdef HEXL_DEBUG
        size_t num_trials = 1;
#else
        size_t num_trials = 10;
#endif
        for (size_t trial = 0; trial < num_trials; ++trial) {
          auto op1 =
              GenerateInsecureUniformRandomValues(length, 0, data_upper_bound);
          auto op2 =
              GenerateInsecureUniformRandomValues(length, 0, data_upper_bound);

          op1[0] = data_upper_bound - 1;
          op2[0] = data_upper_bound - 1;

          switch (input_mod_factor) {
            case 1: {
              EltwiseMultModNative<1>(result_native.data(), op1.data(),
                                      op2.data(), op1.size(), modulus);
              EltwiseMultModAVX512IFMAInt<1>(result_ifma.data(), op1.data(),
                                             op2.data(), op1.size(), modulus);
              break;
            }
            case 2: {
              EltwiseMultModNative<2>(result_native.data(), op1.data(),
                                      op2.data(), op1.size(), modulus);
              EltwiseMultModAVX512IFMAInt<2>(result_ifma.data(), op1.data(),
                                             op2.data(), op1.size(), modulus);
              break;
            }
            case 4: {
              EltwiseMultModNative<4>(result_native.data(), op1.data(),
                                      op2.data(), op1.size(), modulus);
              EltwiseMultModAVX512IFMAInt<4>(result_ifma.data(), op1.data(),
                                             op2.data(), op1.size(), modulus);
            }
          }

          ASSERT_EQ(result_native[0], 1);
          ASSERT_EQ(result_native, result_ifma);
        }
      }
    }
  }
}
#endif

}  // namespace hexl
}  // namespace intel
