// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ntt/ntt-internal.hpp"

#include <cstring>
#include <utility>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/defines.hpp"
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

AllocatorStrategyPtr mallocStrategy = AllocatorStrategyPtr(new MallocStrategy);

NTT::NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity,
         std::shared_ptr<AllocatorBase> alloc_ptr)
    : m_degree(degree),
      m_q(q),
      m_w(root_of_unity),
      m_alloc(alloc_ptr),
      m_aligned_alloc(AlignedAllocator<uint64_t, 64>(m_alloc)),
      m_root_of_unity_powers(m_aligned_alloc),
      m_precon32_root_of_unity_powers(m_aligned_alloc),
      m_precon64_root_of_unity_powers(m_aligned_alloc),
      m_avx512_root_of_unity_powers(m_aligned_alloc),
      m_avx512_precon32_root_of_unity_powers(m_aligned_alloc),
      m_avx512_precon52_root_of_unity_powers(m_aligned_alloc),
      m_avx512_precon64_root_of_unity_powers(m_aligned_alloc),
      m_precon32_inv_root_of_unity_powers(m_aligned_alloc),
      m_precon52_inv_root_of_unity_powers(m_aligned_alloc),
      m_precon64_inv_root_of_unity_powers(m_aligned_alloc),
      m_inv_root_of_unity_powers(m_aligned_alloc) {
  HEXL_CHECK(CheckArguments(degree, q), "");
  HEXL_CHECK(IsPrimitiveRoot(m_w, 2 * degree, q),
             m_w << " is not a primitive 2*" << degree << "'th root of unity");

  m_degree_bits = Log2(m_degree);
  m_w_inv = InverseMod(m_w, m_q);
  ComputeRootOfUnityPowers();
}

NTT::NTT(uint64_t degree, uint64_t q, std::shared_ptr<AllocatorBase> alloc_ptr)
    : NTT(degree, q, MinimalPrimitiveRoot(2 * degree, q), alloc_ptr) {}

void NTT::ComputeRootOfUnityPowers() {
  AlignedVector64<uint64_t> root_of_unity_powers(m_degree, 0, m_aligned_alloc);
  AlignedVector64<uint64_t> inv_root_of_unity_powers(m_degree, 0,
                                                     m_aligned_alloc);

  // 64-bit preconditioned inverse and root of unity powers
  root_of_unity_powers[0] = 1;
  inv_root_of_unity_powers[0] = InverseMod(1, m_q);
  uint64_t idx = 0;
  uint64_t prev_idx = idx;

  for (size_t i = 1; i < m_degree; i++) {
    idx = ReverseBits(i, m_degree_bits);
    root_of_unity_powers[idx] =
        MultiplyMod(root_of_unity_powers[prev_idx], m_w, m_q);
    inv_root_of_unity_powers[idx] = InverseMod(root_of_unity_powers[idx], m_q);

    prev_idx = idx;
  }

  m_root_of_unity_powers = root_of_unity_powers;
  m_avx512_root_of_unity_powers = m_root_of_unity_powers;

  // Duplicate each root of unity at indices [N/4, N/2].
  // These are the roots of unity used in the FwdNTT FwdT2 function
  // By creating these duplicates, we avoid extra permutations while loading the
  // roots of unity
  AlignedVector64<uint64_t> W2_roots;
  W2_roots.reserve(m_degree / 2);
  for (size_t i = m_degree / 4; i < m_degree / 2; ++i) {
    W2_roots.push_back(m_root_of_unity_powers[i]);
    W2_roots.push_back(m_root_of_unity_powers[i]);
  }
  m_avx512_root_of_unity_powers.erase(
      m_avx512_root_of_unity_powers.begin() + m_degree / 4,
      m_avx512_root_of_unity_powers.begin() + m_degree / 2);
  m_avx512_root_of_unity_powers.insert(
      m_avx512_root_of_unity_powers.begin() + m_degree / 4, W2_roots.begin(),
      W2_roots.end());

  // Duplicate each root of unity at indices [N/8, N/4].
  // These are the roots of unity used in the FwdNTT FwdT4 function
  // By creating these duplicates, we avoid extra permutations while loading the
  // roots of unity
  AlignedVector64<uint64_t> W4_roots;
  W4_roots.reserve(m_degree / 2);
  for (size_t i = m_degree / 8; i < m_degree / 4; ++i) {
    W4_roots.push_back(m_root_of_unity_powers[i]);
    W4_roots.push_back(m_root_of_unity_powers[i]);
    W4_roots.push_back(m_root_of_unity_powers[i]);
    W4_roots.push_back(m_root_of_unity_powers[i]);
  }
  m_avx512_root_of_unity_powers.erase(
      m_avx512_root_of_unity_powers.begin() + m_degree / 8,
      m_avx512_root_of_unity_powers.begin() + m_degree / 4);
  m_avx512_root_of_unity_powers.insert(
      m_avx512_root_of_unity_powers.begin() + m_degree / 8, W4_roots.begin(),
      W4_roots.end());

  auto compute_barrett_vector = [&](const AlignedVector64<uint64_t>& values,
                                    uint64_t bit_shift) {
    AlignedVector64<uint64_t> barrett_vector(m_aligned_alloc);
    for (uint64_t value : values) {
      MultiplyFactor mf(value, bit_shift, m_q);
      barrett_vector.push_back(mf.BarrettFactor());
    }
    return barrett_vector;
  };

  m_precon32_root_of_unity_powers =
      compute_barrett_vector(root_of_unity_powers, 32);
  m_precon64_root_of_unity_powers =
      compute_barrett_vector(root_of_unity_powers, 64);

  // 52-bit preconditioned root of unity powers
  if (has_avx512ifma) {
    m_avx512_precon52_root_of_unity_powers =
        compute_barrett_vector(m_avx512_root_of_unity_powers, 52);
  }

  if (has_avx512dq) {
    m_avx512_precon32_root_of_unity_powers =
        compute_barrett_vector(m_avx512_root_of_unity_powers, 32);
    m_avx512_precon64_root_of_unity_powers =
        compute_barrett_vector(m_avx512_root_of_unity_powers, 64);
  }

  // Inverse root of unity powers

  // Reordering inv_root_of_powers
  AlignedVector64<uint64_t> temp(m_degree, 0, m_aligned_alloc);
  temp[0] = inv_root_of_unity_powers[0];
  idx = 1;

  for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
    for (size_t i = 0; i < m; i++) {
      temp[idx] = inv_root_of_unity_powers[m + i];
      idx++;
    }
  }
  m_inv_root_of_unity_powers = std::move(temp);

  // 32-bit preconditioned inverse root of unity powers
  m_precon32_inv_root_of_unity_powers =
      compute_barrett_vector(m_inv_root_of_unity_powers, 32);

  // 52-bit preconditioned inverse root of unity powers
  if (has_avx512ifma) {
    m_precon52_inv_root_of_unity_powers =
        compute_barrett_vector(m_inv_root_of_unity_powers, 52);
  }

  // 64-bit preconditioned inverse root of unity powers
  m_precon64_inv_root_of_unity_powers =
      compute_barrett_vector(m_inv_root_of_unity_powers, 64);
}

bool NTT::CheckArguments(uint64_t degree, uint64_t modulus) {
  HEXL_UNUSED(degree);
  HEXL_UNUSED(modulus);
  HEXL_CHECK(IsPowerOfTwo(degree),
             "degree " << degree << " is not a power of 2");
  HEXL_CHECK(degree <= (1ULL << NTT::MaxDegreeBits()),
             "degree should be less than 2^" << NTT::MaxDegreeBits() << " got "
                                             << degree);
  HEXL_CHECK(modulus <= (1ULL << NTT::MaxModulusBits()),
             "modulus should be less than 2^" << NTT::MaxModulusBits()
                                              << " got " << modulus);
  HEXL_CHECK(modulus % (2 * degree) == 1, "modulus mod 2n != 1");
  HEXL_CHECK(IsPrime(modulus), "modulus is not prime");

  return true;
}

void NTT::ComputeForward(uint64_t* result, const uint64_t* operand,
                         uint64_t input_mod_factor,
                         uint64_t output_mod_factor) {
  HEXL_CHECK(result != nullptr, "result == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "input_mod_factor must be 1, 2 or 4; got " << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
             "output_mod_factor must be 1 or 4; got " << output_mod_factor);
  HEXL_CHECK_BOUNDS(
      operand, m_degree, m_q * input_mod_factor,
      "value in operand exceeds bound " << m_q * input_mod_factor);

#ifdef HEXL_HAS_AVX512IFMA
  if (has_avx512ifma && (m_q < s_max_fwd_ifma_modulus && (m_degree >= 16))) {
    const uint64_t* root_of_unity_powers = GetAVX512RootOfUnityPowers().data();
    const uint64_t* precon_root_of_unity_powers =
        GetAVX512Precon52RootOfUnityPowers().data();

    HEXL_VLOG(3, "Calling 52-bit AVX512-IFMA FwdNTT");
    ForwardTransformToBitReverseAVX512<s_ifma_shift_bits>(
        result, operand, m_degree, m_q, root_of_unity_powers,
        precon_root_of_unity_powers, input_mod_factor, output_mod_factor);
    return;
  }
#endif

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq && m_degree >= 16) {
    if (m_q < s_max_fwd_32_modulus) {
      HEXL_VLOG(3, "Calling 32-bit AVX512-DQ FwdNTT");
      const uint64_t* root_of_unity_powers =
          GetAVX512RootOfUnityPowers().data();
      const uint64_t* precon_root_of_unity_powers =
          GetAVX512Precon32RootOfUnityPowers().data();
      ForwardTransformToBitReverseAVX512<32>(
          result, operand, m_degree, m_q, root_of_unity_powers,
          precon_root_of_unity_powers, input_mod_factor, output_mod_factor);
    } else {
      HEXL_VLOG(3, "Calling 64-bit AVX512-DQ FwdNTT");
      const uint64_t* root_of_unity_powers =
          GetAVX512RootOfUnityPowers().data();
      const uint64_t* precon_root_of_unity_powers =
          GetAVX512Precon64RootOfUnityPowers().data();

      ForwardTransformToBitReverseAVX512<s_default_shift_bits>(
          result, operand, m_degree, m_q, root_of_unity_powers,
          precon_root_of_unity_powers, input_mod_factor, output_mod_factor);
    }
    return;
  }
#endif

  HEXL_VLOG(3, "Calling ForwardTransformToBitReverseRadix2");
  const uint64_t* root_of_unity_powers = GetRootOfUnityPowers().data();
  const uint64_t* precon_root_of_unity_powers =
      GetPrecon64RootOfUnityPowers().data();

  ForwardTransformToBitReverseRadix2(
      result, operand, m_degree, m_q, root_of_unity_powers,
      precon_root_of_unity_powers, input_mod_factor, output_mod_factor);
}

void NTT::ComputeInverse(uint64_t* result, const uint64_t* operand,
                         uint64_t input_mod_factor,
                         uint64_t output_mod_factor) {
  HEXL_CHECK(result != nullptr, "result == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
             "input_mod_factor must be 1 or 2; got " << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2; got " << output_mod_factor);
  HEXL_CHECK_BOUNDS(operand, m_degree, m_q * input_mod_factor,
                    "operand exceeds bound " << m_q * input_mod_factor);

#ifdef HEXL_HAS_AVX512IFMA
  if (has_avx512ifma && (m_q < s_max_inv_ifma_modulus) && (m_degree >= 16)) {
    HEXL_VLOG(3, "Calling 52-bit AVX512-IFMA InvNTT");
    const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowers().data();
    const uint64_t* precon_inv_root_of_unity_powers =
        GetPrecon52InvRootOfUnityPowers().data();
    InverseTransformFromBitReverseAVX512<s_ifma_shift_bits>(
        result, operand, m_degree, m_q, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor);
    return;
  }
#endif

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq && m_degree >= 16) {
    if (m_q < s_max_inv_32_modulus) {
      HEXL_VLOG(3, "Calling 32-bit AVX512-DQ InvNTT");
      const uint64_t* inv_root_of_unity_powers =
          GetInvRootOfUnityPowers().data();
      const uint64_t* precon_inv_root_of_unity_powers =
          GetPrecon32InvRootOfUnityPowers().data();
      InverseTransformFromBitReverseAVX512<32>(
          result, operand, m_degree, m_q, inv_root_of_unity_powers,
          precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor);
    } else {
      HEXL_VLOG(3, "Calling 64-bit AVX512 InvNTT");
      const uint64_t* inv_root_of_unity_powers =
          GetInvRootOfUnityPowers().data();
      const uint64_t* precon_inv_root_of_unity_powers =
          GetPrecon64InvRootOfUnityPowers().data();

      InverseTransformFromBitReverseAVX512<s_default_shift_bits>(
          result, operand, m_degree, m_q, inv_root_of_unity_powers,
          precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor);
    }
    return;
  }
#endif

  HEXL_VLOG(3, "Calling 64-bit default InvNTT");
  const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowers().data();
  const uint64_t* precon_inv_root_of_unity_powers =
      GetPrecon64InvRootOfUnityPowers().data();
  InverseTransformFromBitReverseRadix2(
      result, operand, m_degree, m_q, inv_root_of_unity_powers,
      precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor);
}

}  // namespace hexl
}  // namespace intel
