// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/defines.hpp"
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

  // Inverse transform

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
    HEXL_UNUSED(n);
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
class DegreeModulusInputOutput
    : public ::testing::TestWithParam<std::tuple<
          uint64_t, uint64_t, std::vector<uint64_t>, std::vector<uint64_t>>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

// Test different parts of the public API
TEST_P(DegreeModulusInputOutput, API) {
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

  auto input_radix4 = input;
  InverseTransformFromBitReverseRadix4(
      input_radix4.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
      ntt.GetPrecon64InvRootOfUnityPowers().data(), 2, 1);

  InverseTransformFromBitReverseRadix2(
      input.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
      ntt.GetPrecon64InvRootOfUnityPowers().data(), 2, 1);
}

INSTANTIATE_TEST_SUITE_P(
    NTT, DegreeModulusInputOutput,
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

// First parameter is the NTT degree
// Second parameter is the number of bits in the NTT modulus
class DegreeModulusTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t>> {
 protected:
  void SetUp() {}
  void TearDown() {}

 public:
};

TEST_P(DegreeModulusTest, ForwardZeros) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, true, N)[0];

  std::vector<uint64_t> input(N, 0);
  std::vector<uint64_t> exp_output(N, 0);

  NTT ntt(N, modulus);
  ntt.ComputeForward(input.data(), input.data(), 1, 1);

  AssertEqual(input, exp_output);
}

TEST_P(DegreeModulusTest, InverseZeros) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, true, N)[0];

  std::vector<uint64_t> input(N, 0);
  std::vector<uint64_t> exp_output(N, 0);

  NTT ntt(N, modulus);
  ntt.ComputeInverse(input.data(), input.data(), 1, 1);

  AssertEqual(input, exp_output);
}

TEST_P(DegreeModulusTest, ForwardRadix4Random) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, true, N)[0];

  auto input = GenerateInsecureUniformRandomValues(N, modulus);

  NTT ntt(N, modulus);

  auto input_radix4 = input;
  ForwardTransformToBitReverseRadix4(
      input_radix4.data(), N, modulus, ntt.GetRootOfUnityPowers().data(),
      ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);

  ReferenceForwardTransformToBitReverse(input.data(), N, modulus,
                                        ntt.GetRootOfUnityPowers().data());

  AssertEqual(input, input_radix4);
}

TEST_P(DegreeModulusTest, InverseRadix4Random) {
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, true, N)[0];

  auto input = GenerateInsecureUniformRandomValues(N, modulus);
  auto input_radix4 = input;

  NTT ntt(N, modulus);

  InverseTransformFromBitReverseRadix2(
      input.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
      ntt.GetPrecon64InvRootOfUnityPowers().data(), 2, 1);

  InverseTransformFromBitReverseRadix4(
      input_radix4.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
      ntt.GetPrecon64InvRootOfUnityPowers().data(), 2, 1);

  AssertEqual(input, input_radix4);
}

INSTANTIATE_TEST_SUITE_P(
    NTT, DegreeModulusTest,
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

}  // namespace hexl
}  // namespace intel
