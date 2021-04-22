// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <utility>

#include "hexl/ntt/ntt.hpp"
#include "hexl/util/util.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"
#include "util/check.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

class NTT::NTTImpl {
 public:
  NTTImpl(uint64_t degree, uint64_t q, uint64_t root_of_unity,
          std::shared_ptr<allocator_base> alloc_ptr = {});
  NTTImpl(uint64_t degree, uint64_t q,
          std::shared_ptr<allocator_base> alloc_ptr = {});

  ~NTTImpl();

  uint64_t GetMinimalRootOfUnity() const { return m_w; }

  uint64_t GetDegree() const { return m_degree; }

  uint64_t GetModulus() const { return m_q; }

  AlignedVector64<uint64_t>& GetPrecon64RootOfUnityPowers() {
    return m_precon64_root_of_unity_powers;
  }

  uint64_t* GetPrecon64RootOfUnityPowersPtr() {
    return GetPrecon64RootOfUnityPowers().data();
  }

  AlignedVector64<uint64_t>& GetPrecon52RootOfUnityPowers() {
    return m_precon52_root_of_unity_powers;
  }

  uint64_t* GetPrecon52RootOfUnityPowersPtr() {
    return GetPrecon52RootOfUnityPowers().data();
  }

  uint64_t* GetRootOfUnityPowersPtr() { return GetRootOfUnityPowers().data(); }

  // Returns the vector of pre-computed root of unity powers for the modulus
  // and root of unity.
  AlignedVector64<uint64_t>& GetRootOfUnityPowers() {
    return m_root_of_unity_powers;
  }

  // Returns the root of unity at index i.
  uint64_t GetRootOfUnityPower(size_t i) { return GetRootOfUnityPowers()[i]; }

  // Returns the vector of 64-bit pre-conditioned pre-computed root of unity
  // powers for the modulus and root of unity.
  AlignedVector64<uint64_t>& GetPrecon64InvRootOfUnityPowers() {
    return m_precon64_inv_root_of_unity_powers;
  }

  uint64_t* GetPrecon64InvRootOfUnityPowersPtr() {
    return GetPrecon64InvRootOfUnityPowers().data();
  }

  // Returns the vector of 52-bit pre-conditioned pre-computed root of unity
  // powers for the modulus and root of unity.
  AlignedVector64<uint64_t>& GetPrecon52InvRootOfUnityPowers() {
    return m_precon52_inv_root_of_unity_powers;
  }

  uint64_t* GetPrecon52InvRootOfUnityPowersPtr() {
    return GetPrecon52InvRootOfUnityPowers().data();
  }

  AlignedVector64<uint64_t>& GetInvRootOfUnityPowers() {
    return m_inv_root_of_unity_powers;
  }

  uint64_t* GetInvRootOfUnityPowersPtr() {
    return GetInvRootOfUnityPowers().data();
  }

  uint64_t GetInvRootOfUnityPower(size_t i) {
    return GetInvRootOfUnityPowers()[i];
  }

  void ComputeForward(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

  void ComputeInverse(uint64_t* result, const uint64_t* operand,
                      uint64_t input_mod_factor, uint64_t output_mod_factor);

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
  // Bit shift to use in computing Barrett reduction for forward transform

  uint64_t m_winv;  // Inverse of minimal root of unity
  uint64_t m_w;     // A 2N'th root of unity

  std::shared_ptr<allocator_base> alloc;

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

void ForwardTransformToBitReverse64(uint64_t* operand, uint64_t n,
                                    uint64_t modulus,
                                    const uint64_t* root_of_unity_powers,
                                    const uint64_t* precon_root_of_unity_powers,
                                    uint64_t input_mod_factor = 1,
                                    uint64_t output_mod_factor = 1);

/// @brief Reference NTT which is written for clarity rather than performance
/// @param[in, out] operand Input data. Overwritten with NTT output
/// @param[in] n Size of the transfrom, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] modulus Prime modulus. Must satisfy q == 1 mod 2n
/// @param[in] root_of_unity_powers Powers of 2n'th root of unity in F_q. In
/// bit-reversed order
void ReferenceForwardTransformToBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers);

void InverseTransformFromBitReverse64(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers,
    uint64_t input_mod_factor = 1, uint64_t output_mod_factor = 1);

// Returns true if arguments satisfy constraints for negacyclic NTT
bool CheckNTTArguments(uint64_t degree, uint64_t modulus);

}  // namespace hexl
}  // namespace intel
