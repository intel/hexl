// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "hexl/ntt/ntt.hpp"
#include "logging/logging.hpp"
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#include "ntt/ntt-internal.hpp"
#include "number-theory/number-theory.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(NTT, bad_input) {
  uint64_t p = 769;
  uint64_t N = 8;
  std::vector<uint64_t> input;
  std::vector<uint64_t> p_input;
  std::vector<uint64_t> p_times_2_input;
  std::vector<uint64_t> p_times_4_input;

  NTT ntt(N, p);

  auto init_inputs = [&]() {
    input = {1, 2, 3, 4, 5, 6, 7, 8};
    p_input = std::vector<uint64_t>(N, p);
    p_times_2_input = std::vector<uint64_t>(N, 2 * p);
    p_times_4_input = std::vector<uint64_t>(N, 4 * p);
  };

  // Forward transform
  // Bad input
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeForward(input.data(), nullptr, 1, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeForward(nullptr, input.data(), 1, 1));
  init_inputs();
  EXPECT_NO_THROW(ntt.ComputeForward(input.data(), input.data(), 1, 1));
  init_inputs();
  EXPECT_NO_THROW(ntt.ComputeForward(p_input.data(), p_input.data(), 4, 4));
  init_inputs();
  EXPECT_ANY_THROW(
      ntt.ComputeForward(p_times_2_input.data(), p_times_2_input.data(), 2, 1));
  init_inputs();
  EXPECT_NO_THROW(
      ntt.ComputeForward(p_times_2_input.data(), p_times_2_input.data(), 4, 4));
  init_inputs();
  EXPECT_ANY_THROW(
      ntt.ComputeForward(p_times_4_input.data(), p_times_4_input.data(), 4, 4));
  init_inputs();

  // Bad mod factors
  EXPECT_NO_THROW(ntt.ComputeForward(input.data(), input.data(), 2, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeForward(input.data(), input.data(), 123, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeForward(input.data(), input.data(), 2, 123));
  init_inputs();

  // Inverse tranform

  // Bad input
  EXPECT_ANY_THROW(ntt.ComputeInverse(input.data(), nullptr, 1, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeInverse(nullptr, input.data(), 1, 1));
  init_inputs();

  EXPECT_NO_THROW(ntt.ComputeInverse(input.data(), input.data(), 1, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeInverse(p_input.data(), p_input.data(), 1, 1));
  init_inputs();
  EXPECT_NO_THROW(ntt.ComputeInverse(p_input.data(), p_input.data(), 2, 2));
  init_inputs();
  EXPECT_ANY_THROW(
      ntt.ComputeInverse(p_times_2_input.data(), p_times_2_input.data(), 2, 2));
  init_inputs();

  // Bad mod factors
  EXPECT_NO_THROW(ntt.ComputeInverse(input.data(), input.data(), 1, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeInverse(input.data(), input.data(), 123, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeInverse(input.data(), input.data(), 1, 123));
  init_inputs();
}
#endif

TEST(NTT, Powers) {
  uint64_t modulus = 0xffffffffffc0001ULL;
  {
    uint64_t N = 2;
    NTT::NTTImpl ntt_impl(N, modulus);

    ASSERT_EQ(1ULL, ntt_impl.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt_impl.GetRootOfUnityPower(1));
  }

  {
    uint64_t N = 4;
    NTT::NTTImpl ntt_impl(N, modulus);

    ASSERT_EQ(1ULL, ntt_impl.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt_impl.GetRootOfUnityPower(1));
    ASSERT_EQ(178930308976060547ULL, ntt_impl.GetRootOfUnityPower(2));
    ASSERT_EQ(748001537669050592ULL, ntt_impl.GetRootOfUnityPower(3));
  }
}

TEST(NTT, root_of_unity) {
  uint64_t p = 769;
  uint64_t N = 8;
  std::vector<uint64_t> input{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> input2 = input;

  uint64_t root_of_unity = MinimalPrimitiveRoot(2 * N, p);

  NTT ntt1(N, p);
  NTT ntt2(N, p, root_of_unity);

  ntt1.ComputeForward(input.data(), input.data(), 1, 1);
  ntt2.ComputeForward(input2.data(), input2.data(), 1, 1);

  AssertEqual(input, input2);
}

TEST(NTTImpl, root_of_unity) {
  uint64_t p = 769;
  uint64_t N = 8;

  NTT::NTTImpl ntt_impl(N, p);

  EXPECT_EQ(ntt_impl.GetMinimalRootOfUnity(), MinimalPrimitiveRoot(2 * N, p));
  EXPECT_EQ(ntt_impl.GetDegree(), N);
  EXPECT_EQ(ntt_impl.GetInvRootOfUnityPower(0),
            ntt_impl.GetInvRootOfUnityPowers()[0]);
}

// Parameters = (degree, prime, input, expected_output)
class NTTAPITest
    : public ::testing::TestWithParam<std::tuple<
          uint64_t, uint64_t, std::vector<uint64_t>, std::vector<uint64_t>>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test different parts of the API
TEST_P(NTTAPITest, Fwd) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t prime = std::get<1>(GetParam());

  const std::vector<uint64_t> input_copy = std::get<2>(GetParam());
  std::vector<uint64_t> exp_output = std::get<3>(GetParam());
  std::vector<uint64_t> input = input_copy;
  std::vector<uint64_t> out_buffer(input.size(), 99);

  // In-place Fwd NTT
  NTT::NTTImpl ntt_impl(N, prime);
  NTT ntt(N, prime);
  ntt.ComputeForward(input.data(), input.data(), 1, 1);
  AssertEqual(input, exp_output);

  // In-place lazy NTT
  input = input_copy;
  ntt.ComputeForward(input.data(), input.data(), 2, 4);
  for (auto& elem : input) {
    elem = elem % prime;
  }
  AssertEqual(input, exp_output);

  // Compute reference
  input = input_copy;
  ReferenceForwardTransformToBitReverse(input.data(), N, prime,
                                        ntt_impl.GetRootOfUnityPowers().data());
  AssertEqual(input, exp_output);

  // Test round-trip
  input = input_copy;
  ntt.ComputeForward(out_buffer.data(), input.data(), 1, 1);
  ntt.ComputeInverse(input.data(), out_buffer.data(), 1, 1);
  AssertEqual(input, input_copy);

  // Test out-of-place forward
  input = input_copy;
  ntt.ComputeForward(out_buffer.data(), input.data(), 2, 1);
  AssertEqual(out_buffer, exp_output);

  // Test out-of-place inverse
  input = input_copy;
  ntt.ComputeForward(out_buffer.data(), input.data(), 2, 1);
  ntt.ComputeInverse(input.data(), out_buffer.data(), 1, 1);
  AssertEqual(input, input_copy);

  // Test out-of-place inverse lazy
  input = input_copy;
  ntt.ComputeForward(out_buffer.data(), input.data(), 2, 1);
  ntt.ComputeInverse(input.data(), out_buffer.data(), 1, 2);
  for (auto& elem : input) {
    elem = elem % prime;
  }
  AssertEqual(input, input_copy);
}

INSTANTIATE_TEST_SUITE_P(
    NTTAPITest, NTTAPITest,
    ::testing::Values(
        std::make_tuple(2, 281474976710897, std::vector<uint64_t>{0, 0},
                        std::vector<uint64_t>{0, 0}),
        std::make_tuple(2, 0xffffffffffc0001ULL, std::vector<uint64_t>{0, 0},
                        std::vector<uint64_t>{0, 0}),
        std::make_tuple(2, 281474976710897, std::vector<uint64_t>{1, 0},
                        std::vector<uint64_t>{1, 1}),
        std::make_tuple(2, 281474976710897, std::vector<uint64_t>{1, 1},
                        std::vector<uint64_t>{19842761023586, 261632215687313}),
        std::make_tuple(2, 0xffffffffffc0001ULL, std::vector<uint64_t>{1, 1},
                        std::vector<uint64_t>{288794978602139553,
                                              864126526004445282}),
        std::make_tuple(4, 113, std::vector<uint64_t>{94, 109, 11, 18},
                        std::vector<uint64_t>{82, 2, 81, 98}),
        std::make_tuple(4, 281474976710897,
                        std::vector<uint64_t>{281474976710765, 49,
                                              281474976710643, 275},
                        std::vector<uint64_t>{12006376116355, 216492038983166,
                                              272441922811203, 62009615510542}),
        std::make_tuple(4, 113, std::vector<uint64_t>{59, 50, 98, 50},
                        std::vector<uint64_t>{1, 2, 3, 4}),
        std::make_tuple(4, 73, std::vector<uint64_t>{2, 1, 1, 1},
                        std::vector<uint64_t>{17, 41, 36, 60}),
        std::make_tuple(4, 16417, std::vector<uint64_t>{31, 21, 15, 34},
                        std::vector<uint64_t>{1611, 14407, 14082, 2858}),
        std::make_tuple(4, 4194353,
                        std::vector<uint64_t>{4127, 9647, 1987, 5410},
                        std::vector<uint64_t>{1478161, 3359347, 222964,
                                              3344742}),
        std::make_tuple(8, 4194353,
                        std::vector<uint64_t>{1, 0, 0, 0, 0, 0, 0, 0},
                        std::vector<uint64_t>{1, 1, 1, 1, 1, 1, 1, 1}),
        std::make_tuple(8, 4194353,
                        std::vector<uint64_t>{1, 1, 0, 0, 0, 0, 0, 0},
                        std::vector<uint64_t>{132171, 4062184, 2675172, 1519183,
                                              462763, 3731592, 1824324,
                                              2370031}),
        std::make_tuple(
            32, 769,
            std::vector<uint64_t>{401, 203, 221, 352, 487, 151, 405, 356,
                                  343, 424, 635, 757, 457, 280, 624, 353,
                                  496, 353, 624, 280, 457, 757, 635, 424,
                                  343, 356, 405, 151, 487, 352, 221, 203},
            std::vector<uint64_t>{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32})));

class FwdNTTZerosTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t>> {
 protected:
  void SetUp() {}
  void TearDown() {}

 public:
};

// Parameters = (degree, prime_bits)
TEST_P(FwdNTTZerosTest, Zeros) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t prime_bits = std::get<1>(GetParam());
  uint64_t prime = GeneratePrimes(1, prime_bits, N)[0];

  std::vector<uint64_t> input(N, 0);
  std::vector<uint64_t> exp_output(N, 0);

  NTT ntt(N, prime);
  ntt.ComputeForward(input.data(), input.data(), 1, 1);

  AssertEqual(input, exp_output);
}

INSTANTIATE_TEST_SUITE_P(
    FwdNTTZerosTest, FwdNTTZerosTest,
    ::testing::Values(
        std::make_tuple(1 << 1, 30), std::make_tuple(1 << 2, 30),
        std::make_tuple(1 << 3, 30), std::make_tuple(1 << 4, 35),
        std::make_tuple(1 << 5, 35), std::make_tuple(1 << 6, 35),
        std::make_tuple(1 << 7, 40), std::make_tuple(1 << 8, 40),
        std::make_tuple(1 << 9, 40), std::make_tuple(1 << 10, 45),
        std::make_tuple(1 << 11, 45), std::make_tuple(1 << 12, 45),
        std::make_tuple(1 << 13, 50), std::make_tuple(1 << 14, 50),
        std::make_tuple(1 << 15, 50), std::make_tuple(1 << 16, 55),
        std::make_tuple(1 << 17, 55)));

class InvNTTZerosTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t>> {
 protected:
  void SetUp() {}
  void TearDown() {}

 public:
};

// Parameters = (degree, prime_bits)
TEST_P(InvNTTZerosTest, Zeros) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t prime_bits = std::get<1>(GetParam());
  uint64_t prime = GeneratePrimes(1, prime_bits, N)[0];

  std::vector<uint64_t> input(N, 0);
  std::vector<uint64_t> exp_output(N, 0);

  NTT ntt(N, prime);
  ntt.ComputeInverse(input.data(), input.data(), 1, 1);

  AssertEqual(input, exp_output);
}

INSTANTIATE_TEST_SUITE_P(
    InvNTTZerosTest, InvNTTZerosTest,
    ::testing::Values(
        std::make_tuple(1 << 1, 30), std::make_tuple(1 << 2, 30),
        std::make_tuple(1 << 3, 30), std::make_tuple(1 << 4, 35),
        std::make_tuple(1 << 5, 35), std::make_tuple(1 << 6, 35),
        std::make_tuple(1 << 7, 40), std::make_tuple(1 << 8, 40),
        std::make_tuple(1 << 9, 40), std::make_tuple(1 << 10, 45),
        std::make_tuple(1 << 11, 45), std::make_tuple(1 << 12, 45),
        std::make_tuple(1 << 13, 50), std::make_tuple(1 << 14, 50),
        std::make_tuple(1 << 15, 50), std::make_tuple(1 << 16, 55),
        std::make_tuple(1 << 17, 55)));

#ifdef HEXL_HAS_AVX512IFMA
class NTTPrimesTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test primes around 50 bits to check IFMA behavior
// Parameters = (degree, prime_bits)
TEST_P(NTTPrimesTest, IFMAPrimes) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t prime_bits = std::get<1>(GetParam());
  uint64_t prime = GeneratePrimes(1, prime_bits, N)[0];

  std::vector<uint64_t> input64(N, 0);
  for (size_t i = 0; i < N; ++i) {
    input64[i] = i % prime;
  }
  std::vector<uint64_t> input_ifma = input64;
  std::vector<uint64_t> input_ifma_lazy = input64;

  std::vector<uint64_t> exp_output(N, 0);

  // Compute reference
  NTT::NTTImpl ntt64(N, prime);
  ReferenceForwardTransformToBitReverse(input64.data(), N, prime,
                                        ntt64.GetRootOfUnityPowers().data());

  // Compute with 52-bit bit shift
  NTT::NTTImpl ntt_ifma(N, prime);
  ForwardTransformToBitReverseAVX512<52>(
      input_ifma.data(), N, ntt_ifma.GetModulus(),
      ntt_ifma.GetRootOfUnityPowers().data(),
      ntt_ifma.GetPrecon52RootOfUnityPowers().data(), 2, 1);

  // Compute lazy
  ForwardTransformToBitReverseAVX512<52>(
      input_ifma_lazy.data(), N, ntt_ifma.GetModulus(),
      ntt_ifma.GetRootOfUnityPowers().data(),
      ntt_ifma.GetPrecon52RootOfUnityPowers().data(), 2, 4);
  for (auto& elem : input_ifma_lazy) {
    elem = elem % prime;
  }

  AssertEqual(input64, input_ifma);
  AssertEqual(input64, input_ifma_lazy);
}

INSTANTIATE_TEST_SUITE_P(NTTPrimesTest, NTTPrimesTest,
                         ::testing::Values(std::make_tuple(1 << 4, 48),
                                           std::make_tuple(1 << 5, 49),
                                           std::make_tuple(1 << 6, 49),
                                           std::make_tuple(1 << 7, 49),
                                           std::make_tuple(1 << 8, 49)));
#endif

#ifdef HEXL_HAS_AVX512DQ
TEST(NTT, LoadFwdInterleavedT1) {
  std::vector<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadFwdInterleavedT1(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
  __m512i exp2 = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadInvInterleavedT1) {
  std::vector<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadInvInterleavedT1(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  __m512i exp2 = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadFwdInterleavedT2) {
  std::vector<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadFwdInterleavedT2(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  __m512i exp2 = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadInvInterleavedT2) {
  std::vector<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadInvInterleavedT2(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
  __m512i exp2 = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadFwdInterleavedT4) {
  std::vector<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadFwdInterleavedT4(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
  __m512i exp2 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadInvInterleavedT4) {
  std::vector<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadInvInterleavedT4(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  __m512i exp2 = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, WriteFwdInterleavedT1) {
  std::vector<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  __m512i arg1 = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i arg2 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

  std::vector<uint64_t> out(16, 0);
  std::vector<uint64_t> exp{8,  0, 9,  1, 10, 2, 11, 3,
                            12, 4, 13, 5, 14, 6, 15, 7};

  WriteFwdInterleavedT1(arg1, arg2, reinterpret_cast<__m512i*>(&out[0]));

  AssertEqual(exp, out);
}

TEST(NTT, WriteInvInterleavedT4) {
  std::vector<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                            8, 9, 10, 11, 12, 13, 14, 15};
  __m512i arg1 = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i arg2 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

  std::vector<uint64_t> out(16, 0);
  std::vector<uint64_t> exp{8,  9,  10, 11, 0, 1, 2, 3,
                            12, 13, 14, 15, 4, 5, 6, 7};

  WriteInvInterleavedT4(arg1, arg2, reinterpret_cast<__m512i*>(&out[0]));

  AssertEqual(exp, out);
}

// Checks AVX512 and native forward NTT implementations match
TEST(NTT, FwdNTT_AVX512) {
  uint64_t N = 512;
  uint64_t prime = GeneratePrimes(1, 55, N)[0];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> distrib(0, prime - 1);

  for (size_t trial = 0; trial < 200; ++trial) {
    std::vector<std::uint64_t> input(N, 0);
    for (size_t i = 0; i < N; ++i) {
      input[i] = distrib(gen);
    }
    std::vector<std::uint64_t> input_avx = input;
    std::vector<std::uint64_t> input_avx_lazy = input;

    NTT::NTTImpl ntt_impl(N, prime);
    ForwardTransformToBitReverse64(
        input.data(), N, prime, ntt_impl.GetRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64RootOfUnityPowers().data(), 2, 1);

    ForwardTransformToBitReverseAVX512<64>(
        input_avx.data(), N, ntt_impl.GetModulus(),
        ntt_impl.GetRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64RootOfUnityPowers().data(), 2, 1);

    // Compute lazy
    ForwardTransformToBitReverseAVX512<64>(
        input_avx_lazy.data(), N, ntt_impl.GetModulus(),
        ntt_impl.GetRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64RootOfUnityPowers().data(), 2, 4);
    for (auto& elem : input_avx_lazy) {
      elem = elem % prime;
    }

    ASSERT_EQ(input, input_avx);
    ASSERT_EQ(input, input_avx_lazy);
  }
}

// Checks AVX512 and native InvNTT implementations match
TEST(NTT, InvNTT_AVX512) {
  uint64_t N = 512;
  uint64_t prime = GeneratePrimes(1, 55, N)[0];

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint64_t> distrib(0, prime - 1);

  for (size_t trial = 0; trial < 200; ++trial) {
    std::vector<std::uint64_t> input(N, 0);
    for (size_t i = 0; i < N; ++i) {
      input[i] = distrib(gen);
    }
    std::vector<std::uint64_t> input_avx = input;
    std::vector<std::uint64_t> input_avx_lazy = input;

    NTT::NTTImpl ntt_impl(N, prime);
    InverseTransformFromBitReverse64(
        input.data(), N, prime, ntt_impl.GetInvRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

    InverseTransformFromBitReverseAVX512<64>(
        input_avx.data(), N, ntt_impl.GetModulus(),
        ntt_impl.GetInvRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

    // Compute lazy
    InverseTransformFromBitReverseAVX512<64>(
        input_avx_lazy.data(), N, ntt_impl.GetModulus(),
        ntt_impl.GetInvRootOfUnityPowers().data(),
        ntt_impl.GetPrecon64InvRootOfUnityPowers().data(), 1, 2);
    for (auto& elem : input_avx_lazy) {
      elem = elem % prime;
    }

    ASSERT_EQ(input, input_avx);
    ASSERT_EQ(input, input_avx_lazy);
  }
}
#endif

}  // namespace hexl
}  // namespace intel
