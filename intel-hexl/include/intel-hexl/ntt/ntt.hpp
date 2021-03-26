// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <memory>
#include <vector>

namespace intel {
namespace hexl {

/// @brief Performs negacyclic forward and inverse number-theoretic transform
/// (NTT), commonly used in RLWE cryptography.
/// @details The number-theoretic transform (NTT) specializes the discrete
/// Fourier transform (DFT) to the finite field \f$ \mathbb{Z}_p / (X^N + 1)
/// \f$.
class NTT {
 public:
  /// Initializes an empty NTT object
  NTT();

  /// Destructs the NTT object
  ~NTT();

  /// Initializes an NTT object with degree \p degree and modulus \p p.
  /// @param[in] degree a.k.a. N. Size of the NTT transform. Must be a power of
  /// 2
  /// @param[in] p Prime modulus. Must satisfy \f$ p == 1 \mod 2N \f$
  /// @brief Performs pre-computation necessary for forward and inverse
  /// transforms
  NTT(uint64_t degree, uint64_t p);

  /// @brief Initializes an NTT object with degree \p degree and modulus
  /// \p p.
  /// @param[in] degree a.k.a. N. Size of the NTT transform. Must be a power of
  /// 2
  /// @param[in] p Prime modulus. Must satisfy \f$ p == 1 \mod 2N \f$
  /// @param[in] root_of_unity 2N'th root of unity in \f$ \mathbb{Z_p} \f$.
  /// @details  Performs pre-computation necessary for forward and inverse
  /// transforms
  NTT(uint64_t degree, uint64_t p, uint64_t root_of_unity);

  /// @brief Compute forward NTT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p operand are in [0,
  /// input_mod_factor * p). Must be 1, 2 or 4.
  /// @param[in] output_mod_factor Returns output \p operand in [0,
  /// output_mod_factor * p). Must be 1 or 4.
  void ComputeForward(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  /// Compute inverse NTT. Results are bit-reversed.
  /// @param[out] result Stores the result
  /// @param[in] operand Data on which to compute the NTT
  /// @param[in] input_mod_factor Assume input \p operand are in [0,
  /// input_mod_factor * p). Must be 1 or 2.
  /// @param[in] output_mod_factor Returns output \p operand in [0,
  /// output_mod_factor * p). Must be 1 or 2.
  void ComputeInverse(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  class NTTImpl;  /// Class implementing the NTT

 private:
  std::shared_ptr<NTTImpl> m_impl;
};

}  // namespace hexl
}  // namespace intel
