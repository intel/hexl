// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#include "ntt/ntt-avx512-util.hpp"
#include "ntt/ntt-internal.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(NTT, bad_input) {
  uint64_t N = 8;
  uint64_t modulus = 769;
  std::vector<uint64_t> input;
  std::vector<uint64_t> p_input;
  std::vector<uint64_t> p_times_2_input;
  std::vector<uint64_t> p_times_4_input;

  NTT ntt(N, modulus);

  auto init_inputs = [&]() {
    input = {1, 2, 3, 4, 5, 6, 7, 8};
    p_input = std::vector<uint64_t>(N, modulus);
    p_times_2_input = std::vector<uint64_t>(N, 2 * modulus);
    p_times_4_input = std::vector<uint64_t>(N, 4 * modulus);
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
    NTT ntt(N, modulus);

    ASSERT_EQ(1ULL, ntt.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt.GetRootOfUnityPower(1));
  }

  {
    uint64_t N = 4;
    NTT ntt(N, modulus);

    ASSERT_EQ(1ULL, ntt.GetRootOfUnityPower(0));
    ASSERT_EQ(288794978602139552ULL, ntt.GetRootOfUnityPower(1));
    ASSERT_EQ(178930308976060547ULL, ntt.GetRootOfUnityPower(2));
    ASSERT_EQ(748001537669050592ULL, ntt.GetRootOfUnityPower(3));
  }
}

namespace allocators {
struct CustomAllocator {
  using T = size_t;
  T* invoke_allocation(size_t size) {
    number_allocations++;
    return new T[size];
  }

  void lets_deallocate(T* ptr) {
    number_deallocations++;
    delete[] ptr;
  }
  static size_t number_allocations;
  static size_t number_deallocations;
};

size_t CustomAllocator::number_allocations = 0;
size_t CustomAllocator::number_deallocations = 0;
}  // namespace allocators

template <>
struct NTT::AllocatorAdapter<allocators::CustomAllocator>
    : public AllocatorInterface<
          NTT::AllocatorAdapter<allocators::CustomAllocator>> {
  explicit AllocatorAdapter(allocators::CustomAllocator&& a_)
      : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) {
    return a.invoke_allocation(bytes_count);
  }
  void deallocate(void* p, size_t n) {
    (void)n;
    a.lets_deallocate(static_cast<allocators::CustomAllocator::T*>(p));
  }

  allocators::CustomAllocator a;
};

template <class T>
struct NTT::AllocatorAdapter<std::allocator<T>>
    : public AllocatorInterface<NTT::AllocatorAdapter<std::allocator<T>>> {
  explicit AllocatorAdapter(std::allocator<T>&& a_) : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) { return a.allocate(bytes_count); }
  void deallocate(void* p, size_t n) { a.deallocate(static_cast<T*>(p), n); }

  std::allocator<T> a;
};

TEST(NTT, root_of_unity_with_allocator) {
  uint64_t N = 8;
  uint64_t modulus = 769;
  std::vector<uint64_t> input{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> input2 = input;
  std::vector<uint64_t> input3 = input;
  std::vector<uint64_t> input4 = input;

  uint64_t root_of_unity = MinimalPrimitiveRoot(2 * N, modulus);

  {
    allocators::CustomAllocator a;
    NTT ntt1(N, modulus);
    NTT ntt2(N, modulus, std::move(a));
    NTT ntt3(N, modulus, root_of_unity);

    std::allocator<int> s;
    NTT ntt4(N, modulus, root_of_unity, std::move(s));

    ntt1.ComputeForward(input.data(), input.data(), 1, 1);
    ntt2.ComputeForward(input2.data(), input2.data(), 1, 1);

    ASSERT_NE(allocators::CustomAllocator::number_allocations, 0);

    ntt3.ComputeForward(input3.data(), input3.data(), 1, 1);
    ntt3.ComputeForward(input4.data(), input4.data(), 1, 1);
  }

  ASSERT_NE(allocators::CustomAllocator::number_deallocations, 0);
  AssertEqual(input, input2);
  AssertEqual(input, input3);
  AssertEqual(input, input4);
}

TEST(NTT, root_of_unity) {
  uint64_t N = 8;
  uint64_t modulus = 769;
  std::vector<uint64_t> input{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> input2 = input;

  uint64_t root_of_unity = MinimalPrimitiveRoot(2 * N, modulus);

  NTT ntt1(N, modulus);
  NTT ntt2(N, modulus, root_of_unity);

  ntt1.ComputeForward(input.data(), input.data(), 1, 1);
  ntt2.ComputeForward(input2.data(), input2.data(), 1, 1);

  AssertEqual(input, input2);
}

TEST(NTT, root_of_unity2) {
  uint64_t N = 8;
  uint64_t modulus = 769;

  NTT ntt(N, modulus);

  EXPECT_EQ(ntt.GetMinimalRootOfUnity(), MinimalPrimitiveRoot(2 * N, modulus));
  EXPECT_EQ(ntt.GetDegree(), N);
  EXPECT_EQ(ntt.GetInvRootOfUnityPower(0), ntt.GetInvRootOfUnityPowers()[0]);
}

// Parameters = (degree, modulus, input, expected_output)
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
  uint64_t modulus = std::get<1>(GetParam());

  const std::vector<uint64_t> input_copy = std::get<2>(GetParam());
  std::vector<uint64_t> exp_output = std::get<3>(GetParam());
  std::vector<uint64_t> input = input_copy;
  std::vector<uint64_t> out_buffer(input.size(), 99);

  // In-place Fwd NTT
  NTT ntt(N, modulus);
  ntt.ComputeForward(input.data(), input.data(), 1, 1);
  AssertEqual(input, exp_output);

  // In-place lazy NTT
  input = input_copy;
  ntt.ComputeForward(input.data(), input.data(), 2, 4);
  for (auto& elem : input) {
    elem = elem % modulus;
  }
  AssertEqual(input, exp_output);

  // Compute reference
  input = input_copy;
  ReferenceForwardTransformToBitReverse(input.data(), N, modulus,
                                        ntt.GetRootOfUnityPowers().data());
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
    elem = elem % modulus;
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

// Parameters = (degree, modulus_bits)
TEST_P(FwdNTTZerosTest, Zeros) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, N)[0];

  std::vector<uint64_t> input(N, 0);
  std::vector<uint64_t> exp_output(N, 0);

  NTT ntt(N, modulus);
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

// Parameters = (degree, modulus_bits)
TEST_P(InvNTTZerosTest, Zeros) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, N)[0];

  std::vector<uint64_t> input(N, 0);
  std::vector<uint64_t> exp_output(N, 0);

  NTT ntt(N, modulus);
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
class NTTModulusTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test modulus around 50 bits to check IFMA behavior
// Parameters = (degree, modulus_bits)
TEST_P(NTTModulusTest, IFMAModuli) {
  if (!has_avx512ifma) {
    return;
  }
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, N)[0];

  std::vector<uint64_t> input64(N, 0);
  for (size_t i = 0; i < N; ++i) {
    input64[i] = i % modulus;
  }
  std::vector<uint64_t> input_ifma = input64;
  std::vector<uint64_t> input_ifma_lazy = input64;

  std::vector<uint64_t> exp_output(N, 0);

  // Compute reference
  NTT ntt64(N, modulus);
  ReferenceForwardTransformToBitReverse(input64.data(), N, modulus,
                                        ntt64.GetRootOfUnityPowers().data());

  // Compute with 52-bit bit shift
  NTT ntt_ifma(N, modulus);

  ForwardTransformToBitReverseAVX512<52>(
      input_ifma.data(), N, ntt_ifma.GetModulus(),
      ntt_ifma.GetAVX512RootOfUnityPowers().data(),
      ntt_ifma.GetAVX512Precon52RootOfUnityPowers().data(), 2, 1);

  // Compute lazy
  ForwardTransformToBitReverseAVX512<52>(
      input_ifma_lazy.data(), N, ntt_ifma.GetModulus(),
      ntt_ifma.GetAVX512RootOfUnityPowers().data(),
      ntt_ifma.GetAVX512Precon52RootOfUnityPowers().data(), 2, 4);
  for (auto& elem : input_ifma_lazy) {
    elem = elem % modulus;
  }

  AssertEqual(input64, input_ifma);
  AssertEqual(input64, input_ifma_lazy);
}

INSTANTIATE_TEST_SUITE_P(NTTModulusTest, NTTModulusTest,
                         ::testing::Values(std::make_tuple(1 << 4, 48),
                                           std::make_tuple(1 << 5, 49),
                                           std::make_tuple(1 << 6, 49),
                                           std::make_tuple(1 << 7, 49),
                                           std::make_tuple(1 << 8, 49)));
#endif

#ifdef HEXL_HAS_AVX512DQ
TEST(NTT, LoadFwdInterleavedT1) {
  if (!has_avx512dq) {
    return;
  }

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
  if (!has_avx512dq) {
    return;
  }

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
  if (!has_avx512dq) {
    return;
  }

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
  if (!has_avx512dq) {
    return;
  }

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
  if (!has_avx512dq) {
    return;
  }

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
  if (!has_avx512dq) {
    return;
  }

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
  if (!has_avx512dq) {
    return;
  }

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
  if (!has_avx512dq) {
    return;
  }

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
TEST(NTT, FwdNTT_AVX512_32) {
  if (!has_avx512dq) {
    return;
  }
  std::random_device rd;
  std::mt19937 gen(rd());

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t N = 512; N <= 65536; N *= 2) {
    uint64_t modulus = GeneratePrimes(1, 27, N)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, modulus - 1);

    NTT ntt(N, modulus);

    for (size_t trial = 0; trial < num_trials; ++trial) {
      std::vector<std::uint64_t> input(N, 0);
      for (size_t i = 0; i < N; ++i) {
        input[i] = distrib(gen);
      }
      std::vector<std::uint64_t> input_avx = input;
      std::vector<std::uint64_t> input_avx_lazy = input;

      ForwardTransformToBitReverse64(
          input.data(), N, modulus, ntt.GetRootOfUnityPowers().data(),
          ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);

      ForwardTransformToBitReverseAVX512<32>(
          input_avx.data(), N, ntt.GetModulus(),
          ntt.GetAVX512RootOfUnityPowers().data(),
          ntt.GetAVX512Precon32RootOfUnityPowers().data(), 2, 1);

      // Compute lazy
      ForwardTransformToBitReverseAVX512<32>(
          input_avx_lazy.data(), N, ntt.GetModulus(),
          ntt.GetAVX512RootOfUnityPowers().data(),
          ntt.GetAVX512Precon32RootOfUnityPowers().data(), 2, 4);
      for (auto& elem : input_avx_lazy) {
        elem = elem % modulus;
      }

      ASSERT_EQ(input, input_avx);
      ASSERT_EQ(input, input_avx_lazy);
    }
  }
}

// Checks AVX512 and native forward NTT implementations match
TEST(NTT, FwdNTT_AVX512_64) {
  if (!has_avx512dq) {
    return;
  }
  std::random_device rd;
  std::mt19937 gen(rd());

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t N = 512; N <= 65536; N *= 2) {
    uint64_t modulus = GeneratePrimes(1, 55, N)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, modulus - 1);

    NTT ntt(N, modulus);

    for (size_t trial = 0; trial < num_trials; ++trial) {
      std::vector<std::uint64_t> input(N, 0);
      for (size_t i = 0; i < N; ++i) {
        input[i] = distrib(gen);
      }
      std::vector<std::uint64_t> input_avx = input;
      std::vector<std::uint64_t> input_avx_lazy = input;

      ForwardTransformToBitReverse64(
          input.data(), N, modulus, ntt.GetRootOfUnityPowers().data(),
          ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);

      ForwardTransformToBitReverseAVX512<64>(
          input_avx.data(), N, ntt.GetModulus(),
          ntt.GetAVX512RootOfUnityPowers().data(),
          ntt.GetAVX512Precon64RootOfUnityPowers().data(), 2, 1);

      // Compute lazy
      ForwardTransformToBitReverseAVX512<64>(
          input_avx_lazy.data(), N, ntt.GetModulus(),
          ntt.GetAVX512RootOfUnityPowers().data(),
          ntt.GetAVX512Precon64RootOfUnityPowers().data(), 2, 4);
      for (auto& elem : input_avx_lazy) {
        elem = elem % modulus;
      }

      ASSERT_EQ(input, input_avx);
      ASSERT_EQ(input, input_avx_lazy);
    }
  }
}

// Checks 32-bit AVX512 and native InvNTT implementations match
TEST(NTT, InvNTT_AVX512_32) {
  if (!has_avx512dq) {
    return;
  }
  std::random_device rd;
  std::mt19937 gen(rd());

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t N = 512; N <= 65536; N *= 2) {
    uint64_t modulus = GeneratePrimes(1, 27, N)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, modulus - 1);

    NTT ntt(N, modulus);

    for (size_t trial = 0; trial < num_trials; ++trial) {
      std::vector<std::uint64_t> input(N, 0);
      for (size_t i = 0; i < N; ++i) {
        input[i] = distrib(gen);
      }
      std::vector<std::uint64_t> input_avx = input;
      std::vector<std::uint64_t> input_avx_lazy = input;

      InverseTransformFromBitReverse64(
          input.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

      InverseTransformFromBitReverseAVX512<32>(
          input_avx.data(), N, ntt.GetModulus(),
          ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon32InvRootOfUnityPowers().data(), 1, 1);

      // Compute lazy
      InverseTransformFromBitReverseAVX512<32>(
          input_avx_lazy.data(), N, ntt.GetModulus(),
          ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon32InvRootOfUnityPowers().data(), 1, 2);
      for (auto& elem : input_avx_lazy) {
        elem = elem % modulus;
      }

      ASSERT_EQ(input, input_avx);
      ASSERT_EQ(input, input_avx_lazy);
    }
  }
}

// Checks 64-bit AVX512 and native InvNTT implementations match
TEST(NTT, InvNTT_AVX512_64) {
  if (!has_avx512dq) {
    return;
  }
  std::random_device rd;
  std::mt19937 gen(rd());

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t N = 512; N <= 65536; N *= 2) {
    uint64_t modulus = GeneratePrimes(1, 55, N)[0];
    std::uniform_int_distribution<uint64_t> distrib(0, modulus - 1);

    NTT ntt(N, modulus);

    for (size_t trial = 0; trial < num_trials; ++trial) {
      std::vector<std::uint64_t> input(N, 0);
      for (size_t i = 0; i < N; ++i) {
        input[i] = distrib(gen);
      }
      std::vector<std::uint64_t> input_avx = input;
      std::vector<std::uint64_t> input_avx_lazy = input;

      InverseTransformFromBitReverse64(
          input.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

      InverseTransformFromBitReverseAVX512<64>(
          input_avx.data(), N, ntt.GetModulus(),
          ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

      // Compute lazy
      InverseTransformFromBitReverseAVX512<64>(
          input_avx_lazy.data(), N, ntt.GetModulus(),
          ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 2);
      for (auto& elem : input_avx_lazy) {
        elem = elem % modulus;
      }

      ASSERT_EQ(input, input_avx);
      ASSERT_EQ(input, input_avx_lazy);
    }
  }
}
#endif

}  // namespace hexl
}  // namespace intel
