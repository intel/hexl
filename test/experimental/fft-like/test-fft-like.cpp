// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "hexl/experimental/fft-like/fft-like.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "ntt/ntt-internal.hpp"
#include "test/test-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
TEST(FFTLike, bad_input) {
  uint64_t N = 16;
  double scalar = 1.0;
  AlignedVector64<std::complex<double>> input(N, {0, 0});

  EXPECT_ANY_THROW(FFTLike fft_like(2, nullptr));
  EXPECT_ANY_THROW(FFTLike fft_like(17, nullptr));
  EXPECT_NO_THROW(FFTLike fft_like(16, nullptr));

  FFTLike fft_like(N, nullptr);

  // Forward transform
  // Bad input
  EXPECT_ANY_THROW(
      fft_like.ComputeForwardFFTLike(input.data(), nullptr, &scalar));
  EXPECT_ANY_THROW(
      fft_like.ComputeForwardFFTLike(nullptr, input.data(), &scalar));
  EXPECT_NO_THROW(
      fft_like.ComputeForwardFFTLike(input.data(), input.data(), &scalar));
  EXPECT_NO_THROW(
      fft_like.ComputeForwardFFTLike(input.data(), input.data(), nullptr));

  // Inverse transform
  // Bad input
  EXPECT_ANY_THROW(
      fft_like.ComputeInverseFFTLike(input.data(), nullptr, &scalar));
  EXPECT_ANY_THROW(
      fft_like.ComputeInverseFFTLike(nullptr, input.data(), &scalar));
  EXPECT_NO_THROW(
      fft_like.ComputeInverseFFTLike(input.data(), input.data(), &scalar));
  EXPECT_NO_THROW(
      fft_like.ComputeInverseFFTLike(input.data(), input.data(), nullptr));
}
#endif

TEST(FFTLike, RootsOfUnityNative) {
  {
    FFTLike myfft_like(16, nullptr);
    ASSERT_EQ(std::complex<double>(0, 0), myfft_like.GetComplexRootOfUnity(0));
    ASSERT_EQ(std::complex<double>(-0.38268343236508978, 0.92387953251128674),
              myfft_like.GetComplexRootOfUnity(5));
    ASSERT_EQ(std::complex<double>(0, -1),
              myfft_like.GetInvComplexRootOfUnity(15));
    ASSERT_EQ(std::complex<double>(0.83146961230254524, -0.55557023301960218),
              myfft_like.GetInvComplexRootOfUnity(5));
  }
}

TEST(FFTLike, RootsOfUnityNative2) {
  uint64_t N = 16;

  FFTLike fft_like(N, nullptr);

  EXPECT_EQ(fft_like.GetDegree(), N);
  EXPECT_EQ(fft_like.GetInvComplexRootOfUnity(0),
            fft_like.GetInvComplexRootsOfUnity()[0]);
  EXPECT_EQ(fft_like.GetComplexRootOfUnity(0),
            fft_like.GetComplexRootsOfUnity()[0]);
}

namespace allocators {
struct CustomAllocatorFFTLike {
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

size_t CustomAllocatorFFTLike::number_allocations = 0;
size_t CustomAllocatorFFTLike::number_deallocations = 0;
}  // namespace allocators

template <>
struct FFTLike::AllocatorAdapter<allocators::CustomAllocatorFFTLike>
    : public AllocatorInterface<
          FFTLike::AllocatorAdapter<allocators::CustomAllocatorFFTLike>> {
  explicit AllocatorAdapter(allocators::CustomAllocatorFFTLike&& a_)
      : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) {
    return a.invoke_allocation(bytes_count);
  }
  void deallocate(void* p, size_t n) {
    HEXL_UNUSED(n);
    a.lets_deallocate(static_cast<allocators::CustomAllocatorFFTLike::T*>(p));
  }

  allocators::CustomAllocatorFFTLike a;
};

template <class T>
struct FFTLike::AllocatorAdapter<std::allocator<T>>
    : public AllocatorInterface<FFTLike::AllocatorAdapter<std::allocator<T>>> {
  explicit AllocatorAdapter(std::allocator<T>&& a_) : a(std::move(a_)) {}

  // interface implementations
  void* allocate(size_t bytes_count) { return a.allocate(bytes_count); }
  void deallocate(void* p, size_t n) { a.deallocate(static_cast<T*>(p), n); }

  std::allocator<T> a;
};

TEST(FFTLike, fft_like_with_allocator) {
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
    allocators::CustomAllocatorFFTLike a;
    double scalar = 1 << 16;
    double scale = scalar / static_cast<double>(N);
    double inv_scale = 1.0 / scalar;
    FFTLike fft_like1(N, nullptr);
    FFTLike fft_like2(N, &scalar);
    FFTLike fft_like3(N, &scalar, std::move(a));

    std::allocator<int> s;
    FFTLike fft_like4(N, &scalar, std::move(s));

    fft_like1.ComputeForwardFFTLike(input1.data(), input1.data(), &inv_scale);
    fft_like1.ComputeInverseFFTLike(input1.data(), input1.data(), &scale);
    fft_like2.ComputeForwardFFTLike(input2.data(), input2.data());
    fft_like2.ComputeInverseFFTLike(input2.data(), input2.data());

    ASSERT_NE(allocators::CustomAllocatorFFTLike::number_allocations, 0);

    fft_like3.ComputeForwardFFTLike(input3.data(), input3.data());
    fft_like3.ComputeInverseFFTLike(input3.data(), input3.data());
    fft_like4.ComputeForwardFFTLike(input4.data(), input4.data(), &inv_scale);
    fft_like4.ComputeInverseFFTLike(input4.data(), input4.data(), &scale);
  }

  ASSERT_NE(allocators::CustomAllocatorFFTLike::number_deallocations, 0);
  CheckClose(exp_out, input1, 0.5);
  CheckClose(exp_out, input2, 0.5);
  CheckClose(exp_out, input3, 0.5);
  CheckClose(exp_out, input4, 0.5);
}

}  // namespace hexl
}  // namespace intel
