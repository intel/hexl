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
  /// @param[in] in_scalar Scalar value to calculate scale and inv scale
  /// @param[in] alloc_ptr Custom memory allocator used for intermediate
  /// calculations
  /// @details  Performs pre-computation necessary for forward and inverse
  /// transforms
  FFT(uint64_t degree, double* in_scalar,
      std::shared_ptr<AllocatorBase> alloc_ptr = {});

  template <class Allocator, class... AllocatorArgs>
  FFT(uint64_t degree, double* in_scalar, Allocator&& a,
      AllocatorArgs&&... args)
      : FFT(degree, in_scalar,
            std::static_pointer_cast<AllocatorBase>(
                std::make_shared<AllocatorAdapter<Allocator, AllocatorArgs...>>(
                    std::move(a), std::forward<AllocatorArgs>(args)...))) {}

  /// @brief Compute forward FFT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the FFT
  /// @param[in] in_scale Scale applied to output values
  void ComputeForwardFFT(std::complex<double>* result,
                         const std::complex<double>* operand,
                         const double* in_scale = nullptr);

  /// @brief Compute inverse FFT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the FFT
  /// @param[in] in_scale Scale applied to output values
  void ComputeInverseFFT(std::complex<double>* result,
                         const std::complex<double>* operand,
                         const double* in_scale = nullptr);

  /// @brief Construct floating-point values from CRT-composed polynomial with
  /// integer coefficients.
  /// @param[out] res Stores the result
  /// @param[in] plain Plaintext
  /// @param[in] threshold Upper half threshold with respect to the total
  /// coefficient modulus
  /// @param[in] decryption_modulus Product of all primes in the coefficient
  /// modulus
  /// @param[in] inv_scale Scale applied to output values
  /// @param[in] mod_size Size of coefficient modulus parameter
  /// @param[in] coeff_count Degree of the polynomial modulus parameter
  void BuildFloatingPoints(std::complex<double>* res, const uint64_t* plain,
                           const uint64_t* threshold,
                           const uint64_t* decryption_modulus,
                           const double inv_scale, size_t mod_size,
                           size_t coeff_count);

  /// @brief Returns the root of unity power at bit-reversed index i.
  /// @param[in] i Index
  std::complex<double> GetComplexRootOfUnity(size_t i) {
    return GetComplexRootsOfUnity()[i];
  }

  /// @brief Returns the root of unity in bit-reversed order
  const AlignedVector64<std::complex<double>>& GetComplexRootsOfUnity() const {
    return m_complex_roots_of_unity;
  }

  /// @brief Returns the root of unity power at bit-reversed index i.
  /// @param[in] i Index
  std::complex<double> GetInvComplexRootOfUnity(size_t i) {
    return GetInvComplexRootsOfUnity()[i];
  }

  /// @brief Returns the inverse root of unity in bit-reversed order
  const AlignedVector64<std::complex<double>>& GetInvComplexRootsOfUnity()
      const {
    return m_inv_complex_roots_of_unity;
  }

  /// @brief Returns the degree N
  uint64_t GetDegree() const { return m_degree; }

 private:
  // Computes 1~(n-1)-th powers and inv powers of the primitive 2n-th root
  void ComputeComplexRootsOfUnity();

  uint64_t m_degree;  // N: size of FFT transform, should be power of 2

  double* scalar;  // Pointer to scalar used for scale/inv_scale calculation

  double scale;  // Scale value use for encoding (inv fft)

  double inv_scale;  // Scale value use in decoding (fwd fft)

  std::shared_ptr<AllocatorBase> m_alloc;

  AlignedAllocator<double, 64> m_aligned_alloc;

  uint64_t m_degree_bits;  // log_2(m_degree)

  // Contains 0~(n-1)-th powers of the 2n-th primitive root.
  AlignedVector64<std::complex<double>> m_complex_roots_of_unity;

  // Contains 0~(n-1)-th inv powers of the 2n-th primitive inv root.
  AlignedVector64<std::complex<double>> m_inv_complex_roots_of_unity;
};

}  // namespace hexl
}  // namespace intel
