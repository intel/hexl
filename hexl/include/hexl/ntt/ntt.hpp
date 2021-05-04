// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <memory>
#include <vector>

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
  /// @param[in] output_mod_factor Returns output \p operand in [0,
  /// output_mod_factor * q). Must be 1 or 4.
  void ComputeForward(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  /// Compute inverse NTT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p operand are in [0,
  /// input_mod_factor * q). Must be 1 or 2.
  /// @param[in] output_mod_factor Returns output \p operand in [0,
  /// output_mod_factor * q). Must be 1 or 2.
  void ComputeInverse(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  class NTTImpl;  /// Class implementing the NTT

 private:
  std::shared_ptr<NTTImpl> m_impl;
};

}  // namespace hexl
}  // namespace intel
