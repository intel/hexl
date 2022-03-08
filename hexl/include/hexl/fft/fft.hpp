// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>

#include "hexl/fft/fwd-fft-avx512.hpp"
#include "hexl/fft/inv-fft-avx512.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/allocator.hpp"

namespace intel {
namespace hexl {

/// @brief Performs linear forward and inverse FFT-like transform
/// for CKKS encoding and decoding.
class FFT {
 public:
  /// @brief Helper class for custom memory allocation
  template <class Adaptee, class... Args>
  struct AllocatorAdapter
      : public AllocatorInterface<AllocatorAdapter<Adaptee, Args...>> {
    explicit AllocatorAdapter(Adaptee&& _a, Args&&... args);
    AllocatorAdapter(const Adaptee& _a, Args&... args);

    // interface implementation
    void* allocate_impl(size_t bytes_count);
    void deallocate_impl(void* p, size_t n);

   private:
    Adaptee alloc;
  };

  /// @brief Initializes an empty CKKS_FTT object
  FFT() = default;

  /// @brief Destructs the CKKS_FTT object
  ~FFT() = default;

  /// @brief Initializes an FFT object with degree \p degree and scalar
  /// \p in_scalar.
  /// @param[in] degree also known as N. Size of the FFT transform. Must be a
  /// power of 2
  /// @param[in] alloc_ptr Custom memory allocator used for intermediate
  /// calculations
  /// @details  Performs pre-computation necessary for forward and inverse
  /// transforms
  explicit FFT(uint64_t degree, std::shared_ptr<AllocatorBase> alloc_ptr = {});

  template <class Allocator, class... AllocatorArgs>
  FFT(uint64_t degree, Allocator&& a, AllocatorArgs&&... args)
      : FFT(degree,
            std::static_pointer_cast<AllocatorBase>(
                std::make_shared<AllocatorAdapter<Allocator, AllocatorArgs...>>(
                    std::move(a), std::forward<AllocatorArgs>(args)...))) {}

  /// @brief Compute forward FFT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the FFT
  void ComputeForwardFFT(std::complex<double>* result,
                         const std::complex<double>* operand);

  /// @brief Compute inverse FFT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the FFT
  void ComputeInverseFFT(std::complex<double>* result,
                         const std::complex<double>* operand);

  /// @brief Returns the root of unity power at index i.
  /// @param[in] i Index
  std::complex<double> GetComplexRootOfUnity(size_t i) {
    return GetComplexRootsOfUnity()[i];
  }

  /// @brief Returns the roots of unity
  const AlignedVector64<std::complex<double>>& GetComplexRootsOfUnity() const {
    return m_complex_roots_of_unity;
  }

  /// @brief Returns the interleaved roots of unity
  const AlignedVector64<double>& GetInterleavedComplexRootsOfUnity() const {
    return m_interleaved_complex_roots_of_unity;
  }

  /// @brief Returns the root of unity power at index i
  /// @param[in] i Index
  std::complex<double> GetInvComplexRootOfUnity(size_t i) {
    return GetInvComplexRootsOfUnity()[i];
  }

  /// @brief Returns the inverse roots of unity
  const AlignedVector64<std::complex<double>>& GetInvComplexRootsOfUnity()
      const {
    return m_inv_complex_roots_of_unity;
  }

  /// @brief Returns the interleaved inverse roots of unity
  const AlignedVector64<double>& GetInterleavedInvComplexRootsOfUnity() const {
    return m_interleaved_inv_complex_roots_of_unity;
  }

  /// @brief Returns the degree N
  uint64_t GetDegree() const { return m_degree; }

 private:
  // Computes 1~(n-1)-th powers and inv powers of the primitive 2n-th root
  void ComputeComplexRootsOfUnity();

  uint64_t m_degree;  // N: size of FFT transform, should be power of 2

  std::shared_ptr<AllocatorBase> m_alloc;

  AlignedAllocator<double, 64> m_aligned_alloc;

  uint64_t m_degree_bits;  // log_2(m_degree)

  // Contains 0~(n-1)-th powers of the 2n-th primitive root.
  AlignedVector64<std::complex<double>> m_complex_roots_of_unity;
  AlignedVector64<double> m_interleaved_complex_roots_of_unity;

  // Contains 0~(n-1)-th inv powers of the 2n-th primitive inv root.
  AlignedVector64<std::complex<double>> m_inv_complex_roots_of_unity;
  AlignedVector64<double> m_interleaved_inv_complex_roots_of_unity;
};

}  // namespace hexl
}  // namespace intel
