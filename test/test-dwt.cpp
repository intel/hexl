// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "hexl/dwt/dwt.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(DWT, bad_input) {
  uint64_t N = 16;
  double scalar = 1.0;
  AlignedVector64<std::complex<double>> input(N, {0, 0});

  EXPECT_ANY_THROW(DWT dwt(2, nullptr));
  EXPECT_ANY_THROW(DWT dwt(17, nullptr));
  EXPECT_NO_THROW(DWT dwt(16, nullptr));

  DWT dwt(N, nullptr);

  // Forward transform
  // Bad input
  EXPECT_ANY_THROW(dwt.ComputeForwardDWT(input.data(), nullptr, &scalar));
  EXPECT_ANY_THROW(dwt.ComputeForwardDWT(nullptr, input.data(), &scalar));
  EXPECT_NO_THROW(dwt.ComputeForwardDWT(input.data(), input.data(), &scalar));
  EXPECT_NO_THROW(dwt.ComputeForwardDWT(input.data(), input.data(), nullptr));

  // Inverse transform
  // Bad input
  EXPECT_ANY_THROW(dwt.ComputeInverseDWT(input.data(), nullptr, &scalar));
  EXPECT_ANY_THROW(dwt.ComputeInverseDWT(nullptr, input.data(), &scalar));
  EXPECT_NO_THROW(dwt.ComputeInverseDWT(input.data(), input.data(), &scalar));
  EXPECT_NO_THROW(dwt.ComputeInverseDWT(input.data(), input.data(), nullptr));
}
#endif

TEST(DWT, RootsOfUnityNative) {
  {
    DWT mydwt(16, nullptr);
    ASSERT_EQ(std::complex<double>(0, 0), mydwt.GetComplexRootOfUnity(0));
    ASSERT_EQ(std::complex<double>(-0.38268343236508978, 0.92387953251128674),
              mydwt.GetComplexRootOfUnity(5));
    ASSERT_EQ(std::complex<double>(0, -1), mydwt.GetInvComplexRootOfUnity(15));
    ASSERT_EQ(std::complex<double>(0.83146961230254524, -0.55557023301960218),
              mydwt.GetInvComplexRootOfUnity(5));
  }
}

TEST(DWT, RootsOfUnityNative2) {
  uint64_t N = 16;

  DWT dwt(N, nullptr);

  EXPECT_EQ(dwt.GetDegree(), N);
  EXPECT_EQ(dwt.GetInvComplexRootOfUnity(0),
            dwt.GetInvComplexRootsOfUnity()[0]);
  EXPECT_EQ(dwt.GetComplexRootOfUnity(0), dwt.GetComplexRootsOfUnity()[0]);
}

namespace allocators {
struct CustomAllocatorDWT {
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

size_t CustomAllocatorDWT::number_allocations = 0;
size_t CustomAllocatorDWT::number_deallocations = 0;
}  // namespace allocators

template <>
struct DWT::AllocatorAdapter<allocators::CustomAllocatorDWT>
    : public AllocatorInterface<
          DWT::AllocatorAdapter<allocators::CustomAllocatorDWT>> {
  explicit AllocatorAdapter(allocators::CustomAllocatorDWT&& a_)
      : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) {
    return a.invoke_allocation(bytes_count);
  }
  void deallocate(void* p, size_t n) {
    HEXL_UNUSED(n);
    a.lets_deallocate(static_cast<allocators::CustomAllocatorDWT::T*>(p));
  }

  allocators::CustomAllocatorDWT a;
};

template <class T>
struct DWT::AllocatorAdapter<std::allocator<T>>
    : public AllocatorInterface<DWT::AllocatorAdapter<std::allocator<T>>> {
  explicit AllocatorAdapter(std::allocator<T>&& a_) : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) { return a.allocate(bytes_count); }
  void deallocate(void* p, size_t n) { a.deallocate(static_cast<T*>(p), n); }

  std::allocator<T> a;
};

TEST(DWT, dwt_with_allocator) {
  uint64_t N = 16;
  const double data_bound = (1 << 30);
  AlignedVector64<std::complex<double>> input1(N);
  for (size_t i = 0; i < N; i++) {
    input1[i] = std::complex<double>(
        GenerateInsecureUniformRealRandomValue(0, data_bound),
        GenerateInsecureUniformRealRandomValue(0, data_bound));
  }
  AlignedVector64<std::complex<double>> input2 = input1;
  AlignedVector64<std::complex<double>> input3 = input1;
  AlignedVector64<std::complex<double>> input4 = input1;
  AlignedVector64<std::complex<double>> exp_out = input1;

  {
    allocators::CustomAllocatorDWT a;
    double scalar = 1 << 16;
    double scale = scalar / static_cast<double>(N);
    double inv_scale = 1.0 / scalar;
    DWT dwt1(N, nullptr);
    DWT dwt2(N, &scalar);
    DWT dwt3(N, &scalar, std::move(a));

    std::allocator<int> s;
    DWT dwt4(N, &scalar, std::move(s));

    dwt1.ComputeForwardDWT(input1.data(), input1.data(), &inv_scale);
    dwt1.ComputeInverseDWT(input1.data(), input1.data(), &scale);
    dwt2.ComputeForwardDWT(input2.data(), input2.data());
    dwt2.ComputeInverseDWT(input2.data(), input2.data());

    ASSERT_NE(allocators::CustomAllocatorDWT::number_allocations, 0);

    dwt3.ComputeForwardDWT(input3.data(), input3.data());
    dwt3.ComputeInverseDWT(input3.data(), input3.data());
    dwt4.ComputeForwardDWT(input4.data(), input4.data(), &inv_scale);
    dwt4.ComputeInverseDWT(input4.data(), input4.data(), &scale);
  }

  ASSERT_NE(allocators::CustomAllocatorDWT::number_deallocations, 0);
  CheckClose(exp_out, input1, 0.5);
  CheckClose(exp_out, input2, 0.5);
  CheckClose(exp_out, input3, 0.5);
  CheckClose(exp_out, input4, 0.5);
}

}  // namespace hexl
}  // namespace intel
