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

/// @brief Performs negacyclic forward and inverse number-theoretic transform
/// (NTT), commonly used in RLWE cryptography.
/// @details The number-theoretic transform (NTT) specializes the discrete
/// Fourier transform (DFT) to the finite field \f$ \mathbb{Z}_q[X] / (X^N + 1)
/// \f$.
class NTT {
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

  /// @brief Initializes an empty NTT object
  NTT() = default;

  /// @brief Destructs the NTT object
  ~NTT() = default;

  /// @brief Initializes an NTT object with degree \p degree and modulus \p q.
  /// @param[in] degree also known as N. Size of the NTT transform. Must be a
  /// power of
  /// 2
  /// @param[in] q Prime modulus. Must satisfy \f$ q == 1 \mod 2N \f$
  /// @param[in] alloc_ptr Custom memory allocator used for intermediate
  /// calculations
  /// @brief Performs pre-computation necessary for forward and inverse
  /// transforms
  NTT(uint64_t degree, uint64_t q,
      std::shared_ptr<AllocatorBase> alloc_ptr = {});

  template <class Allocator, class... AllocatorArgs>
  NTT(uint64_t degree, uint64_t q, Allocator&& a, AllocatorArgs&&... args)
      : NTT(degree, q,
            std::static_pointer_cast<AllocatorBase>(
                std::make_shared<AllocatorAdapter<Allocator, AllocatorArgs...>>(
                    std::move(a), std::forward<AllocatorArgs>(args)...))) {}

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
  NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity,
      std::shared_ptr<AllocatorBase> alloc_ptr = {});

  template <class Allocator, class... AllocatorArgs>
  NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity, Allocator&& a,
      AllocatorArgs&&... args)
      : NTT(degree, q, root_of_unity,
            std::static_pointer_cast<AllocatorBase>(
                std::make_shared<AllocatorAdapter<Allocator, AllocatorArgs...>>(
                    std::move(a), std::forward<AllocatorArgs>(args)...))) {}

  /// @brief Returns true if arguments satisfy constraints for negacyclic NTT
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
  void ComputeForward(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  /// Compute inverse NTT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p operand are in [0,
  /// input_mod_factor * q). Must be 1 or 2.
  /// @param[in] output_mod_factor Returns output \p result in [0,
  /// output_mod_factor * q). Must be 1 or 2.
  void ComputeInverse(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  /// @brief Returns the minimal 2N'th root of unity
  uint64_t GetMinimalRootOfUnity() const { return m_w; }

  /// @brief Returns the degree N
  uint64_t GetDegree() const { return m_degree; }

  /// @brief Returns the word-sized prime modulus
  uint64_t GetModulus() const { return m_q; }

  /// @brief Returns the root of unity powers in bit-reversed order
  const AlignedVector64<uint64_t>& GetRootOfUnityPowers() const {
    return m_root_of_unity_powers;
  }

  /// @brief Returns the root of unity power at bit-reversed index i.
  uint64_t GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

  /// @brief Returns 32-bit pre-conditioned root of unity powers in
  /// bit-reversed order
  const AlignedVector64<uint64_t>& GetPrecon32RootOfUnityPowers() const {
    return m_precon32_root_of_unity_powers;
  }

  /// @brief Returns 64-bit pre-conditioned root of unity powers in
  /// bit-reversed order
  const AlignedVector64<uint64_t>& GetPrecon64RootOfUnityPowers() const {
    return m_precon64_root_of_unity_powers;
  }

  /// @brief Returns the root of unity powers in bit-reversed order with
  /// modifications for use by AVX512 implementation
  const AlignedVector64<uint64_t>& GetAVX512RootOfUnityPowers() const {
    return m_avx512_root_of_unity_powers;
  }

  /// @brief Returns 32-bit pre-conditioned AVX512 root of unity powers in
  /// bit-reversed order
  const AlignedVector64<uint64_t>& GetAVX512Precon32RootOfUnityPowers() const {
    return m_avx512_precon32_root_of_unity_powers;
  }

  /// @brief Returns 52-bit pre-conditioned AVX512 root of unity powers in
  /// bit-reversed order
  const AlignedVector64<uint64_t>& GetAVX512Precon52RootOfUnityPowers() const {
    return m_avx512_precon52_root_of_unity_powers;
  }

  /// @brief Returns 64-bit pre-conditioned AVX512 root of unity powers in
  /// bit-reversed order
  const AlignedVector64<uint64_t>& GetAVX512Precon64RootOfUnityPowers() const {
    return m_avx512_precon64_root_of_unity_powers;
  }

  /// @brief Returns the inverse root of unity powers in bit-reversed order
  const AlignedVector64<uint64_t>& GetInvRootOfUnityPowers() const {
    return m_inv_root_of_unity_powers;
  }

  /// @brief Returns the inverse root of unity power at bit-reversed index i.
  uint64_t GetInvRootOfUnityPower(size_t i) {
    return GetInvRootOfUnityPowers()[i];
  }

  /// @brief Returns the vector of 32-bit pre-conditioned pre-computed root of
  /// unity
  // powers for the modulus and root of unity.
  const AlignedVector64<uint64_t>& GetPrecon32InvRootOfUnityPowers() const {
    return m_precon32_inv_root_of_unity_powers;
  }

  /// @brief Returns the vector of 52-bit pre-conditioned pre-computed root of
  /// unity
  // powers for the modulus and root of unity.
  const AlignedVector64<uint64_t>& GetPrecon52InvRootOfUnityPowers() const {
    return m_precon52_inv_root_of_unity_powers;
  }

  /// @brief Returns the vector of 64-bit pre-conditioned pre-computed root of
  /// unity
  // powers for the modulus and root of unity.
  const AlignedVector64<uint64_t>& GetPrecon64InvRootOfUnityPowers() const {
    return m_precon64_inv_root_of_unity_powers;
  }

  /// @brief Maximum power of 2 in degree
  static size_t MaxDegreeBits() { return 20; }

  /// @brief Maximum number of bits in modulus;
  static size_t MaxModulusBits() { return 62; }

  /// @brief Default bit shift used in Barrett precomputation
  static const size_t s_default_shift_bits{64};

  /// @brief Bit shift used in Barrett precomputation when AVX512-IFMA
  /// acceleration is enabled
  static const size_t s_ifma_shift_bits{52};

  /// @brief Maximum modulus to use 32-bit AVX512-DQ acceleration for the
  /// forward transform
  static const size_t s_max_fwd_32_modulus{1ULL << (32 - 2)};

  /// @brief Maximum modulus to use 32-bit AVX512-DQ acceleration for the
  /// inverse transform
  static const size_t s_max_inv_32_modulus{1ULL << (32 - 2)};

  /// @brief Maximum modulus to use AVX512-IFMA acceleration for the forward
  /// transform
  static const size_t s_max_fwd_ifma_modulus{1ULL << (s_ifma_shift_bits - 2)};

  /// @brief Maximum modulus to use AVX512-IFMA acceleration for the inverse
  /// transform
  static const size_t s_max_inv_ifma_modulus{1ULL << (s_ifma_shift_bits - 2)};

  /// @brief Maximum modulus to use AVX512-DQ acceleration for the inverse
  /// transform
  static const size_t s_max_inv_dq_modulus{1ULL << (s_default_shift_bits - 2)};

  static size_t s_max_fwd_modulus(int bit_shift) {
    if (bit_shift == 32) {
      return s_max_fwd_32_modulus;
    } else if (bit_shift == 52) {
      return s_max_fwd_ifma_modulus;
    } else if (bit_shift == 64) {
      return 1ULL << MaxModulusBits();
    }
    HEXL_CHECK(false, "Invalid bit_shift " << bit_shift);
    return 0;
  }

  static size_t s_max_inv_modulus(int bit_shift) {
    if (bit_shift == 32) {
      return s_max_inv_32_modulus;
    } else if (bit_shift == 52) {
      return s_max_inv_ifma_modulus;
    } else if (bit_shift == 64) {
      return 1ULL << MaxModulusBits();
    }
    HEXL_CHECK(false, "Invalid bit_shift " << bit_shift);
    return 0;
  }

 private:
  void ComputeRootOfUnityPowers();

  uint64_t m_degree;  // N: size of NTT transform, should be power of 2
  uint64_t m_q;       // prime modulus. Must satisfy q == 1 mod 2n

  uint64_t m_degree_bits;  // log_2(m_degree)

  uint64_t m_w_inv;  // Inverse of minimal root of unity
  uint64_t m_w;      // A 2N'th root of unity

  std::shared_ptr<AllocatorBase> m_alloc;

  AlignedAllocator<uint64_t, 64> m_aligned_alloc;

  // powers of the minimal root of unity
  AlignedVector64<uint64_t> m_root_of_unity_powers;
  // vector of floor(W * 2**32 / m_q), with W the root of unity powers
  AlignedVector64<uint64_t> m_precon32_root_of_unity_powers;
  // vector of floor(W * 2**64 / m_q), with W the root of unity powers
  AlignedVector64<uint64_t> m_precon64_root_of_unity_powers;

  // powers of the minimal root of unity adjusted for use in AVX512
  // implementations
  AlignedVector64<uint64_t> m_avx512_root_of_unity_powers;
  // vector of floor(W * 2**32 / m_q), with W the AVX512 root of unity powers
  AlignedVector64<uint64_t> m_avx512_precon32_root_of_unity_powers;
  // vector of floor(W * 2**52 / m_q), with W the AVX512 root of unity powers
  AlignedVector64<uint64_t> m_avx512_precon52_root_of_unity_powers;
  // vector of floor(W * 2**64 / m_q), with W the AVX512 root of unity powers
  AlignedVector64<uint64_t> m_avx512_precon64_root_of_unity_powers;

  // vector of floor(W * 2**32 / m_q), with W the inverse root of unity powers
  AlignedVector64<uint64_t> m_precon32_inv_root_of_unity_powers;
  // vector of floor(W * 2**52 / m_q), with W the inverse root of unity powers
  AlignedVector64<uint64_t> m_precon52_inv_root_of_unity_powers;
  // vector of floor(W * 2**64 / m_q), with W the inverse root of unity powers
  AlignedVector64<uint64_t> m_precon64_inv_root_of_unity_powers;

  AlignedVector64<uint64_t> m_inv_root_of_unity_powers;
};

}  // namespace hexl
}  // namespace intel
