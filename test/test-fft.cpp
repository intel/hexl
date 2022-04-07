// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "hexl/fft/fft.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(FFT, bad_input) {
  uint64_t N = 16;
  AlignedVector64<std::complex<double>> input(N, {0, 0});

  EXPECT_ANY_THROW(FFT fft(2));
  EXPECT_ANY_THROW(FFT fft(17));
  EXPECT_NO_THROW(FFT fft(16));

  FFT fft(N);

  // Forward transform
  // Bad input
  EXPECT_ANY_THROW(fft.ComputeForwardFFT(input.data(), nullptr));
  EXPECT_ANY_THROW(fft.ComputeForwardFFT(nullptr, input.data()));
  EXPECT_NO_THROW(fft.ComputeForwardFFT(input.data(), input.data()));

  // Inverse transform
  // Bad input
  EXPECT_ANY_THROW(fft.ComputeInverseFFT(input.data(), nullptr));
  EXPECT_ANY_THROW(fft.ComputeInverseFFT(nullptr, input.data()));
  EXPECT_NO_THROW(fft.ComputeInverseFFT(input.data(), input.data()));
}
#endif

TEST(FFT, RootsOfUnityNative) {
  {
    FFT myfft(16);
    ASSERT_EQ(std::complex<double>(0, 0), myfft.GetComplexRootOfUnity(0));
    ASSERT_EQ(std::complex<double>(-0.38268343236508978, 0.92387953251128674),
              myfft.GetComplexRootOfUnity(5));
    ASSERT_EQ(std::complex<double>(0, -1), myfft.GetInvComplexRootOfUnity(15));
    ASSERT_EQ(std::complex<double>(0.83146961230254524, -0.55557023301960218),
              myfft.GetInvComplexRootOfUnity(5));
  }
}

TEST(FFT, RootsOfUnityNative2) {
  uint64_t N = 16;

  FFT fft(N);

  EXPECT_EQ(fft.GetDegree(), N);
  EXPECT_EQ(fft.GetInvComplexRootOfUnity(0),
            fft.GetInvComplexRootsOfUnity()[0]);
  EXPECT_EQ(fft.GetComplexRootOfUnity(0), fft.GetComplexRootsOfUnity()[0]);
}

namespace allocators {
struct CustomAllocatorFFT {
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

size_t CustomAllocatorFFT::number_allocations = 0;
size_t CustomAllocatorFFT::number_deallocations = 0;
}  // namespace allocators

template <>
struct FFT::AllocatorAdapter<allocators::CustomAllocatorFFT>
    : public AllocatorInterface<
          FFT::AllocatorAdapter<allocators::CustomAllocatorFFT>> {
  explicit AllocatorAdapter(allocators::CustomAllocatorFFT&& a_)
      : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) {
    return a.invoke_allocation(bytes_count);
  }
  void deallocate(void* p, size_t n) {
    HEXL_UNUSED(n);
    a.lets_deallocate(static_cast<allocators::CustomAllocatorFFT::T*>(p));
  }

  allocators::CustomAllocatorFFT a;
};

template <class T>
struct FFT::AllocatorAdapter<std::allocator<T>>
    : public AllocatorInterface<FFT::AllocatorAdapter<std::allocator<T>>> {
  explicit AllocatorAdapter(std::allocator<T>&& a_) : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) { return a.allocate(bytes_count); }
  void deallocate(void* p, size_t n) { a.deallocate(static_cast<T*>(p), n); }

  std::allocator<T> a;
};

TEST(FFT, fft_with_allocator) {
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
  AlignedVector64<std::complex<double>> exp_out = input1;

  {
    allocators::CustomAllocatorFFT a;
    FFT fft1(N);
    FFT fft2(N, std::move(a));

    std::allocator<int> s;
    FFT fft3(N, std::move(s));

    fft1.ComputeForwardFFT(input1.data(), input1.data());
    fft1.ComputeInverseFFT(input1.data(), input1.data());

    ASSERT_NE(allocators::CustomAllocatorFFT::number_allocations, 0);

    fft2.ComputeForwardFFT(input2.data(), input2.data());
    fft2.ComputeInverseFFT(input2.data(), input2.data());
    fft3.ComputeForwardFFT(input3.data(), input3.data());
    fft3.ComputeInverseFFT(input3.data(), input3.data());
  }

  ASSERT_NE(allocators::CustomAllocatorFFT::number_deallocations, 0);
  CheckClose(exp_out, input1, 0.5);
  CheckClose(exp_out, input2, 0.5);
  CheckClose(exp_out, input3, 0.5);
}

}  // namespace hexl
}  // namespace intel
