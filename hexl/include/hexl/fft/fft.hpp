// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <memory>
#include <vector>

#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/allocator.hpp"

namespace intel {
namespace hexl {

/// @brief Performs linear forward and inverse FTT-like transform
/// for CKKS encodnig and decoding.
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

  /// @brief Initializes an NTT object with degree \p degree and modulus
  /// \p q.
  /// @param[in] degree also known as N. Size of the NTT transform. Must be a
  /// power of
  /// 2
  /// @param[in] q Prime modulus. Must satisfy \f$ q == 1 \mod 2N \f$
  /// @param[in] root_of_unity 2N'th root of unity in \f$ \mathbb{Z_q} \f$.
  /// @param[in] alloc_ptr Custom memory allocator used for intermediate
  /// calculations
  /// @details  Performs pre-computation necessary for forward and inverse
  /// transforms
  FFT(uint64_t degree, double_t* roots_of_unity_real,
      double_t* roots_of_unity_imag, double_t scalar,
      std::shared_ptr<AllocatorBase> alloc_ptr = {});

  template <class Allocator, class... AllocatorArgs>
  FFT(uint64_t degree, double_t* roots_of_unity_real,
      double_t* roots_of_unity_imag, double_t scalar, Allocator&& a,
      AllocatorArgs&&... args)
      : FFT(degree, roots_of_unity_real, roots_of_unity_imag, scalar,
            std::static_pointer_cast<AllocatorBase>(
                std::make_shared<AllocatorAdapter<Allocator, AllocatorArgs...>>(
                    std::move(a), std::forward<AllocatorArgs>(args)...))) {}

  /// @brief Returns true if arguments satisfy constraints for FTT-like
  /// transform
  /// @param[in] degree N. Size of the transform, i.e. the polynomial degree.
  /// Must be a power of two.
  /// @param[in] modulus Prime modulus q. Must satisfy q mod 2N = 1
  static bool CheckArguments(uint64_t degree, uint64_t modulus);

  /// @brief Compute forward NTT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p operand are in [0,
  /// input_mod_factor * q). Must be 1, 2 or 4.
  /// @param[in] output_mod_factor Returns output \p result in [0,
  /// output_mod_factor * q). Must be 1 or 4.
  void ComputeForwardFFT(double_t* result_real, double_t* result_imag,
                         const double_t* operand_real,
                         const double_t* operand_imag);

 private:
  // Primitive 2n-th root, m = 2n
  void ComputeComplexPrimitiveRootOfUnityPowers();

  // 1~(n-1)-th powers of the primitive 2n-th root
  void ComputeComplexRootOfUnityPowers();

  static constexpr double PI_ = 3.1415926535897932384626433832795028842;

  double_t scalar;  //

  uint64_t m_degree;  // N: size of FFT transform, should be power of 2

  uint64_t m_degree_bits;  // log_2(m_degree)

  // Contains 0~(n/8-1)-th powers of the n-th primitive root.
  // AlignedVector64<uint64_t> m_root_of_unity_powers_real;
  AlignedVector64<double_t> m_complex_root_of_unity_powers_real;
  AlignedVector64<double_t> m_complex_root_of_unity_powers_imag;
};

}  // namespace hexl
}  // namespace intel
