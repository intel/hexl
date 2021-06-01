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

  /// Initializes an empty NTT object
  NTT();

  /// Destructs the NTT object
  ~NTT();

  /// Initializes an NTT object with degree \p degree and modulus \p q.
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

  uint64_t GetMinimalRootOfUnity() const { return m_w; }

  uint64_t GetDegree() const { return m_degree; }

  uint64_t GetModulus() const { return m_q; }

  const AlignedVector64<uint64_t>& GetPrecon64RootOfUnityPowers() const {
    return m_precon64_root_of_unity_powers;
  }

  const AlignedVector64<uint64_t>& GetPrecon52RootOfUnityPowers() const {
    return m_precon52_root_of_unity_powers;
  }

  // Returns the vector of pre-computed root of unity powers for the modulus
  // and root of unity.
  const AlignedVector64<uint64_t>& GetRootOfUnityPowers() const {
    return m_root_of_unity_powers;
  }

  // Returns the root of unity at index i.
  uint64_t GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

  // Returns the vector of 64-bit pre-conditioned pre-computed root of unity
  // powers for the modulus and root of unity.
  const AlignedVector64<uint64_t>& GetPrecon64InvRootOfUnityPowers() const {
    return m_precon64_inv_root_of_unity_powers;
  }

  // Returns the vector of 52-bit pre-conditioned pre-computed root of unity
  // powers for the modulus and root of unity.
  const AlignedVector64<uint64_t>& GetPrecon52InvRootOfUnityPowers() const {
    return m_precon52_inv_root_of_unity_powers;
  }

  const AlignedVector64<uint64_t>& GetInvRootOfUnityPowers() const {
    return m_inv_root_of_unity_powers;
  }

  uint64_t GetInvRootOfUnityPower(size_t i) {
    return GetInvRootOfUnityPowers()[i];
  }

  static const size_t s_max_degree_bits{20};  // Maximum power of 2 in degree

  // Maximum number of bits in modulus;
  static const size_t s_max_modulus_bits{62};

  // Default bit shift used in Barrett precomputation
  static const size_t s_default_shift_bits{64};

  // Bit shift used in Barrett precomputation when IFMA acceleration is enabled
  static const size_t s_ifma_shift_bits{52};

  // Maximum number of bits in modulus to use IFMA acceleration for the forward
  // transform
  static const size_t s_max_fwd_ifma_modulus{1ULL << (s_ifma_shift_bits - 2)};

  // Maximum number of bits in modulus to use IFMA acceleration for the inverse
  // transform
  static const size_t s_max_inv_ifma_modulus{1ULL << (s_ifma_shift_bits - 1)};

 private:
  void ComputeRootOfUnityPowers();

  uint64_t m_degree;  // N: size of NTT transform, should be power of 2
  uint64_t m_q;       // prime modulus. Must satisfy q == 1 mod 2n

  uint64_t m_degree_bits;  // log_2(m_degree)

  uint64_t m_winv;  // Inverse of minimal root of unity
  uint64_t m_w;     // A 2N'th root of unity

  std::shared_ptr<AllocatorBase> m_alloc;

  AlignedAllocator<uint64_t, 64> m_aligned_alloc;

  // vector of floor(W * 2**52 / m_q), with W the root of unity powers
  AlignedVector64<uint64_t> m_precon52_root_of_unity_powers;
  // vector of floor(W * 2**64 / m_q), with W the root of unity powers
  AlignedVector64<uint64_t> m_precon64_root_of_unity_powers;
  // powers of the minimal root of unity
  AlignedVector64<uint64_t> m_root_of_unity_powers;

  // vector of floor(W * 2**52 / m_q), with W the inverse root of unity powers
  AlignedVector64<uint64_t> m_precon52_inv_root_of_unity_powers;
  // vector of floor(W * 2**64 / m_q), with W the inverse root of unity powers
  AlignedVector64<uint64_t> m_precon64_inv_root_of_unity_powers;

  AlignedVector64<uint64_t> m_inv_root_of_unity_powers;
};

}  // namespace hexl
}  // namespace intel
