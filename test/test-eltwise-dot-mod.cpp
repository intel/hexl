// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "eltwise/eltwise-dot-mod-avx512.hpp"
#include "eltwise/eltwise-dot-mod-internal.hpp"
#include "hexl/eltwise/eltwise-dot-mod.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

// Parameters = (input, operand1, operand2 operand3, operand4, n, modulus)
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

  uint64_t num_vectors = 2;
  std::vector<uint64_t> output(n, 0);

  std::vector<const uint64_t*> x_addr{&operand1[0], &operand3[0]};
  std::vector<const uint64_t*> y_addr{&operand2[0], &operand4[0]};

  EltwiseDotModNative(output.data(), x_addr.data(), y_addr.data(), num_vectors,
                      n, modulus);

  CheckEqual(expected, output);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseDotModTest, EltwiseDotModTest,
    ::testing::Values(
        std::make_tuple(std::vector<uint64_t>{34, 88, 46, 8, 74, 44, 18, 96},
                        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8},
                        std::vector<uint64_t>{9, 10, 11, 12, 13, 14, 15, 16},
                        std::vector<uint64_t>{17, 18, 19, 20, 21, 22, 23, 24},
                        std::vector<uint64_t>{25, 26, 27, 28, 29, 30, 31, 32},
                        8, 100),
        std::make_tuple(
            std::vector<uint64_t>{192098826, 379819053, 645134975, 446213836,
                                  726368534, 897110544, 169192354, 828516840},
            std::vector<uint64_t>{320011846, 853979704, 1000277200, 995523962,
                                  493907702, 875195366, 238730711, 189327928},
            std::vector<uint64_t>{78001059, 294981723, 813979507, 425467254,
                                  1005007643, 94501959, 652104838, 448934817},
            std::vector<uint64_t>{210626960, 737578009, 518620094, 801559987,
                                  813139663, 916111331, 569708812, 716992044},
            std::vector<uint64_t>{420336371, 851066432, 474525696, 182075850,
                                  919981410, 81415718, 397775612, 519163048},
            8, 1073730817)));

class EltwiseDotModTestAVX512
    : public ::testing::TestWithParam<std::tuple<
          std::vector<uint64_t>, std::vector<uint64_t>, std::vector<uint64_t>,
          std::vector<uint64_t>, std::vector<uint64_t>, uint64_t, uint64_t>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test AVX512 implementation
TEST_P(EltwiseDotModTestAVX512, AVX512) {
  std::vector<uint64_t> expected = std::get<0>(GetParam());
  std::vector<uint64_t> operand1 = std::get<1>(GetParam());
  std::vector<uint64_t> operand2 = std::get<2>(GetParam());
  std::vector<uint64_t> operand3 = std::get<3>(GetParam());
  std::vector<uint64_t> operand4 = std::get<4>(GetParam());
  uint64_t n = std::get<5>(GetParam());
  uint64_t modulus = std::get<6>(GetParam());

  uint64_t num_vectors = 2;
  std::vector<uint64_t> output(n, 0);

  std::vector<const uint64_t*> x_addr{&operand1[0], &operand3[0]};
  std::vector<const uint64_t*> y_addr{&operand2[0], &operand4[0]};

  EltwiseDotModAVX512(output.data(), x_addr.data(), y_addr.data(), num_vectors,
                      n, modulus);

  CheckEqual(expected, output);
}

INSTANTIATE_TEST_SUITE_P(
    EltwiseDotModTestAVX512, EltwiseDotModTestAVX512,
    ::testing::Values(
        std::make_tuple(std::vector<uint64_t>{34, 88, 46, 8, 74, 44, 18, 96},
                        std::vector<uint64_t>{1, 2, 3, 4, 5, 6, 7, 8},
                        std::vector<uint64_t>{9, 10, 11, 12, 13, 14, 15, 16},
                        std::vector<uint64_t>{17, 18, 19, 20, 21, 22, 23, 24},
                        std::vector<uint64_t>{25, 26, 27, 28, 29, 30, 31, 32},
                        8, 100),
        std::make_tuple(
            std::vector<uint64_t>{192098826, 379819053, 645134975, 446213836,
                                  726368534, 897110544, 169192354, 828516840},
            std::vector<uint64_t>{320011846, 853979704, 1000277200, 995523962,
                                  493907702, 875195366, 238730711, 189327928},
            std::vector<uint64_t>{78001059, 294981723, 813979507, 425467254,
                                  1005007643, 94501959, 652104838, 448934817},
            std::vector<uint64_t>{210626960, 737578009, 518620094, 801559987,
                                  813139663, 916111331, 569708812, 716992044},
            std::vector<uint64_t>{420336371, 851066432, 474525696, 182075850,
                                  919981410, 81415718, 397775612, 519163048},
            8, 1073730817)));

// Checks AVX512 and native eltwise add implementations match
// #ifdef HEXL_HAS_AVX512DQ
// TEST(EltwiseDotMod, avx512_native_match) {
//   std::random_device rd;
//   std::mt19937 gen(42);  // (rd());

//   size_t length = 8;

//   for (size_t bits = 10; bits <= 12; ++bits) {
//     uint64_t modulus = 1ULL << bits;

//     std::uniform_int_distribution<uint64_t> distrib(0, modulus - 1);

// #ifdef HEXL_DEBUG
//     size_t num_trials = 1;
// #else
//     size_t num_trials = 1;
// #endif

//     for (size_t trial = 0; trial < num_trials; ++trial) {
//       std::vector<uint64_t> op1(length, 0);
//       std::vector<uint64_t> op2(length, 0);
//       std::vector<uint64_t> op3(length, 0);
//       std::vector<uint64_t> op4(length, 0);
//       std::vector<uint64_t> out_native(length, 0);
//       std::vector<uint64_t> out_avx(length, 0);
//       for (size_t i = 0; i < length; ++i) {
//         op1[i] = distrib(gen);
//         op2[i] = distrib(gen);
//         op3[i] = distrib(gen);
//         op4[i] = distrib(gen);
//       }

//       EltwiseDotModNative(out_native.data(), op1.data(), op2.data(),
//       op3.data(),
//                           op4.data(), length, modulus);
//       EltwiseDotModAVX512(out_avx.data(), op1.data(), op2.data(), op3.data(),
//                           op4.data(), length, modulus);

//       ASSERT_EQ(out_native, out_avx);
//     }
//   }
// }

// #endif

}  // namespace hexl
}  // namespace intel
