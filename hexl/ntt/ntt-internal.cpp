// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ntt/ntt-internal.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <utility>

#include "logging/logging.hpp"
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#include "number-theory/number-theory.hpp"
#include "util/aligned-allocator.hpp"
#include "util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

allocator_strategy_ptr mallocStrategy =
    allocator_strategy_ptr(new details::MallocStrategy);

NTT::NTTImpl::NTTImpl(uint64_t degree, uint64_t q, uint64_t root_of_unity,
                      std::shared_ptr<allocator_base> alloc_ptr)
    : m_degree(degree),
      m_q(q),
      m_w(root_of_unity),
      alloc(alloc_ptr),
      m_precon52_root_of_unity_powers(alloc),
      m_precon64_root_of_unity_powers(alloc),
      m_root_of_unity_powers(alloc),
      m_precon52_inv_root_of_unity_powers(alloc),
      m_precon64_inv_root_of_unity_powers(alloc),
      m_inv_root_of_unity_powers(alloc) {
  alloc = alloc_ptr;

  HEXL_CHECK(CheckNTTArguments(degree, q), "");
  HEXL_CHECK(IsPrimitiveRoot(m_w, 2 * degree, q),
             m_w << " is not a primitive 2*" << degree << "'th root of unity");

#ifdef HEXL_HAS_AVX512IFMA
  if (m_q < s_max_fwd_ifma_modulus) {
    HEXL_VLOG(3, "Setting m_fwd_bit_shift to " << s_ifma_shift_bits);
    m_fwd_bit_shift = s_ifma_shift_bits;
  }
  if (m_q < s_max_inv_ifma_modulus) {
    HEXL_VLOG(3, "Setting m_inv_bit_shift to " << s_ifma_shift_bits);
    m_inv_bit_shift = s_ifma_shift_bits;
  }
#endif

  m_degree_bits = Log2(m_degree);
  m_winv = InverseUIntMod(m_w, m_q);
  ComputeRootOfUnityPowers();
}

NTT::NTTImpl::NTTImpl(uint64_t degree, uint64_t q,
                      std::shared_ptr<allocator_base> alloc_ptr)
    : NTTImpl(degree, q, MinimalPrimitiveRoot(2 * degree, q), alloc_ptr) {}

NTT::NTTImpl::~NTTImpl() = default;

void NTT::NTTImpl::ComputeRootOfUnityPowers() {
  AlignedVector64<uint64_t> root_of_unity_powers(m_degree, 0, alloc);
  AlignedVector64<uint64_t> inv_root_of_unity_powers(m_degree, 0, alloc);

  // 64-bit  precon
  root_of_unity_powers[0] = 1;
  inv_root_of_unity_powers[0] = InverseUIntMod(1, m_q);
  uint64_t idx = 0;
  uint64_t prev_idx = idx;

  for (size_t i = 1; i < m_degree; i++) {
    idx = ReverseBitsUInt(i, m_degree_bits);
    root_of_unity_powers[idx] =
        MultiplyUIntMod(root_of_unity_powers[prev_idx], m_w, m_q);
    inv_root_of_unity_powers[idx] =
        InverseUIntMod(root_of_unity_powers[idx], m_q);

    prev_idx = idx;
  }

  // Reordering inv_root_of_powers
  AlignedVector64<uint64_t> temp(m_degree, 0, alloc);
  temp[0] = inv_root_of_unity_powers[0];
  idx = 1;

  for (size_t m = (m_degree >> 1); m > 0; m >>= 1) {
    for (size_t i = 0; i < m; i++) {
      temp[idx] = inv_root_of_unity_powers[m + i];
      idx++;
    }
  }
  inv_root_of_unity_powers = temp;

  // 64-bit preconditioned root of unity powers
  AlignedVector64<uint64_t> precon64_root_of_unity_powers(alloc);
  precon64_root_of_unity_powers.reserve(m_degree);
  for (uint64_t root_of_unity : root_of_unity_powers) {
    MultiplyFactor mf(root_of_unity, 64, m_q);
    precon64_root_of_unity_powers.push_back(mf.BarrettFactor());
  }

  NTT::NTTImpl::GetPrecon64RootOfUnityPowers() =
      std::move(precon64_root_of_unity_powers);

  // 52-bit preconditioned root of unity powers
  AlignedVector64<uint64_t> precon52_root_of_unity_powers(alloc);
  precon52_root_of_unity_powers.reserve(m_degree);
  for (uint64_t root_of_unity : root_of_unity_powers) {
    MultiplyFactor mf(root_of_unity, 52, m_q);
    precon52_root_of_unity_powers.push_back(mf.BarrettFactor());
  }

  NTT::NTTImpl::GetPrecon52RootOfUnityPowers() =
      std::move(precon52_root_of_unity_powers);

  NTT::NTTImpl::GetRootOfUnityPowers() = std::move(root_of_unity_powers);

  // 64-bit preconditioned inverse root of unity powers
  AlignedVector64<uint64_t> precon64_inv_root_of_unity_powers(alloc);
  precon64_inv_root_of_unity_powers.reserve(m_degree);
  for (uint64_t inv_root_of_unity : inv_root_of_unity_powers) {
    MultiplyFactor mf(inv_root_of_unity, 64, m_q);
    precon64_inv_root_of_unity_powers.push_back(mf.BarrettFactor());
  }

  NTT::NTTImpl::GetPrecon64InvRootOfUnityPowers() =
      std::move(precon64_inv_root_of_unity_powers);

  // 52-bit preconditioned inverse root of unity powers
  AlignedVector64<uint64_t> precon52_inv_root_of_unity_powers(alloc);
  precon52_inv_root_of_unity_powers.reserve(m_degree);
  for (uint64_t inv_root_of_unity : inv_root_of_unity_powers) {
    MultiplyFactor mf(inv_root_of_unity, 52, m_q);
    precon52_inv_root_of_unity_powers.push_back(mf.BarrettFactor());
  }

  NTT::NTTImpl::GetPrecon52InvRootOfUnityPowers() =
      std::move(precon52_inv_root_of_unity_powers);

  NTT::NTTImpl::GetInvRootOfUnityPowers() = std::move(inv_root_of_unity_powers);
}

void NTT::NTTImpl::ComputeForward(uint64_t* result, const uint64_t* operand,
                                  uint64_t input_mod_factor,
                                  uint64_t output_mod_factor) {
  HEXL_CHECK(m_fwd_bit_shift == s_ifma_shift_bits ||
                 m_fwd_bit_shift == s_default_shift_bits,
             "Bit shift " << m_fwd_bit_shift << " should be either "
                          << s_ifma_shift_bits << " or "
                          << s_default_shift_bits);
  HEXL_CHECK(result != nullptr, "result == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK_BOUNDS(
      operand, m_degree, m_q * input_mod_factor,
      "value in operand exceeds bound " << m_q * input_mod_factor);

  if (result != operand) {
    std::memcpy(result, operand, m_degree * sizeof(uint64_t));
  }

#ifdef HEXL_HAS_AVX512IFMA
  if (has_avx512ifma && m_fwd_bit_shift == s_ifma_shift_bits &&
      (m_q < s_max_fwd_ifma_modulus && (m_degree >= 16))) {
    const uint64_t* root_of_unity_powers = GetRootOfUnityPowersPtr();
    const uint64_t* precon_root_of_unity_powers =
        GetPrecon52RootOfUnityPowersPtr();

    HEXL_VLOG(3, "Calling 52-bit AVX512-IFMA NTT");
    ForwardTransformToBitReverseAVX512<s_ifma_shift_bits>(
        result, m_degree, m_q, root_of_unity_powers,
        precon_root_of_unity_powers, input_mod_factor, output_mod_factor);
    return;
  }
#endif

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq && m_degree >= 16) {
    HEXL_VLOG(3, "Calling 64-bit AVX512 NTT");
    const uint64_t* root_of_unity_powers = GetRootOfUnityPowersPtr();
    const uint64_t* precon_root_of_unity_powers =
        GetPrecon64RootOfUnityPowersPtr();

    ForwardTransformToBitReverseAVX512<s_default_shift_bits>(
        result, m_degree, m_q, root_of_unity_powers,
        precon_root_of_unity_powers, input_mod_factor, output_mod_factor);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling 64-bit default NTT");
  const uint64_t* root_of_unity_powers = GetRootOfUnityPowersPtr();
  const uint64_t* precon_root_of_unity_powers =
      GetPrecon64RootOfUnityPowersPtr();

  ForwardTransformToBitReverse64(result, m_degree, m_q, root_of_unity_powers,
                                 precon_root_of_unity_powers, input_mod_factor,
                                 output_mod_factor);
}

void NTT::NTTImpl::ComputeInverse(uint64_t* result, const uint64_t* operand,
                                  uint64_t input_mod_factor,
                                  uint64_t output_mod_factor) {
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");

  HEXL_CHECK_BOUNDS(operand, m_degree, m_q * input_mod_factor,
                    "operand exceeds bound " << m_q * input_mod_factor);

  if (operand != result) {
    std::memcpy(result, operand, m_degree * sizeof(uint64_t));
  }

  HEXL_CHECK(m_inv_bit_shift == s_ifma_shift_bits ||
                 m_inv_bit_shift == s_default_shift_bits,
             "Bit shift " << m_inv_bit_shift << " should be either "
                          << s_ifma_shift_bits << " or "
                          << s_default_shift_bits);

#ifdef HEXL_HAS_AVX512IFMA
  if (has_avx512ifma && m_inv_bit_shift == s_ifma_shift_bits &&
      (m_q < s_max_inv_ifma_modulus) && (m_degree >= 16)) {
    HEXL_VLOG(3, "Calling 52-bit AVX512-IFMA InvNTT");
    const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowersPtr();
    const uint64_t* precon_inv_root_of_unity_powers =
        GetPrecon52InvRootOfUnityPowersPtr();
    InverseTransformFromBitReverseAVX512<s_ifma_shift_bits>(
        result, m_degree, m_q, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor);
    return;
  }
#endif

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq && m_degree >= 16) {
    HEXL_VLOG(3, "Calling 64-bit AVX512 InvNTT");
    const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowersPtr();
    const uint64_t* precon_inv_root_of_unity_powers =
        GetPrecon64InvRootOfUnityPowersPtr();

    InverseTransformFromBitReverseAVX512<s_default_shift_bits>(
        result, m_degree, m_q, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor);
    return;
  }
#endif

  HEXL_VLOG(3, "Calling 64-bit default InvNTT");
  const uint64_t* inv_root_of_unity_powers = GetInvRootOfUnityPowersPtr();
  const uint64_t* precon_inv_root_of_unity_powers =
      GetPrecon64InvRootOfUnityPowersPtr();
  InverseTransformFromBitReverse64(
      result, m_degree, m_q, inv_root_of_unity_powers,
      precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor);
}

// NTT API
NTT::NTT() = default;

NTT::NTT(uint64_t degree, uint64_t q,
         std::shared_ptr<allocator_base> alloc_ptr /* = {}*/)
    : m_impl(std::make_shared<NTT::NTTImpl>(degree, q, alloc_ptr)) {}

NTT::NTT(uint64_t degree, uint64_t q, uint64_t root_of_unity,
         std::shared_ptr<allocator_base> alloc_ptr /* = {}*/)
    : m_impl(std::make_shared<NTT::NTTImpl>(degree, q, root_of_unity,
                                            alloc_ptr)) {}

NTT::~NTT() = default;

void NTT::ComputeForward(uint64_t* result, const uint64_t* operand,
                         uint64_t input_mod_factor,
                         uint64_t output_mod_factor) {
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(result != nullptr, "result == nullptr");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "input_mod_factor must be 1, 2 or 4; got " << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
             "output_mod_factor must be 1 or 4; got " << output_mod_factor);

  m_impl->ComputeForward(result, operand, input_mod_factor, output_mod_factor);
}

void NTT::ComputeInverse(uint64_t* result, const uint64_t* operand,
                         uint64_t input_mod_factor,
                         uint64_t output_mod_factor) {
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(result != nullptr, "result == nullptr");
  HEXL_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
             "input_mod_factor must be 1 or 2; got " << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2; got " << output_mod_factor);

  m_impl->ComputeInverse(result, operand, input_mod_factor, output_mod_factor);
}

// Free functions

void ForwardTransformToBitReverse64(uint64_t* operand, uint64_t n,
                                    uint64_t modulus,
                                    const uint64_t* root_of_unity_powers,
                                    const uint64_t* precon_root_of_unity_powers,
                                    uint64_t input_mod_factor,
                                    uint64_t output_mod_factor) {
  HEXL_CHECK(CheckNTTArguments(n, modulus), "");
  HEXL_CHECK_BOUNDS(operand, n, modulus * input_mod_factor,
                    "operand exceeds bound " << modulus * input_mod_factor);
  HEXL_CHECK(root_of_unity_powers != nullptr,
             "root_of_unity_powers == nullptr");
  HEXL_CHECK(precon_root_of_unity_powers != nullptr,
             "precon_root_of_unity_powers == nullptr");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "input_mod_factor must be 1, 2, or 4; got " << input_mod_factor);
  (void)(input_mod_factor);  // Avoid unused parameter warning
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
             "output_mod_factor must be 1 or 4; got " << output_mod_factor);

  uint64_t twice_mod = modulus << 1;
  size_t t = (n >> 1);

  for (size_t m = 1; m < n; m <<= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = root_of_unity_powers[m + i];
      const uint64_t W_precon = precon_root_of_unity_powers[m + i];

      uint64_t* X = operand + j1;
      uint64_t* Y = X + t;

      uint64_t tx;
      uint64_t T;
      HEXL_LOOP_UNROLL_4
      for (size_t j = j1; j < j2; j++) {
        // The Harvey butterfly: assume X, Y in [0, 4q), and return X', Y'
        // in [0, 4q). Such that X', Y' = X + WY, X - WY (mod q).
        // See Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf
        HEXL_CHECK(*X < modulus * 4, "input X " << (*X) << " too large");
        HEXL_CHECK(*Y < modulus * 4, "input Y " << (*Y) << " too large");

        tx = (*X >= twice_mod) ? (*X - twice_mod) : *X;
        T = MultiplyUIntModLazy<64>(*Y, W_op, W_precon, modulus);

        *X++ = tx + T;
        *Y++ = tx + twice_mod - T;

        HEXL_CHECK(tx + T < modulus * 4,
                   "ouput X " << (tx + T) << " too large");
        HEXL_CHECK(tx + twice_mod - T < modulus * 4,
                   "output Y " << (tx + twice_mod - T) << " too large");
      }
      j1 += (t << 1);
    }
    t >>= 1;
  }
  if (output_mod_factor == 1) {
    for (size_t i = 0; i < n; ++i) {
      if (operand[i] >= twice_mod) {
        operand[i] -= twice_mod;
      }
      if (operand[i] >= modulus) {
        operand[i] -= modulus;
      }
      HEXL_CHECK(operand[i] < modulus, "Incorrect modulus reduction in NTT "
                                           << operand[i] << " >= " << modulus);
    }
  }
}

void ReferenceForwardTransformToBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers) {
  HEXL_CHECK(CheckNTTArguments(n, modulus), "");
  HEXL_CHECK(root_of_unity_powers != nullptr,
             "root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");

  size_t t = (n >> 1);
  for (size_t m = 1; m < n; m <<= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = root_of_unity_powers[m + i];

      uint64_t* X = operand + j1;
      uint64_t* Y = X + t;
      for (size_t j = j1; j < j2; j++) {
        uint64_t tx = *X;
        // X', Y' = X + WY, X - WY (mod q).
        uint64_t W_x_Y = MultiplyUIntMod(*Y, W_op, modulus);
        *X++ = AddUIntMod(tx, W_x_Y, modulus);
        *Y++ = SubUIntMod(tx, W_x_Y, modulus);
      }
      j1 += (t << 1);
    }
    t >>= 1;
  }
}

void InverseTransformFromBitReverse64(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor) {
  HEXL_CHECK(CheckNTTArguments(n, modulus), "");
  HEXL_CHECK(inv_root_of_unity_powers != nullptr,
             "inv_root_of_unity_powers == nullptr");
  HEXL_CHECK(precon_inv_root_of_unity_powers != nullptr,
             "precon_inv_root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
             "input_mod_factor must be 1 or 2; got " << input_mod_factor);
  (void)(input_mod_factor);  // Avoid unused parameter warning
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2; got " << output_mod_factor);

  uint64_t twice_mod = modulus << 1;
  size_t t = 1;
  size_t root_index = 1;

  for (size_t m = (n >> 1); m > 1; m >>= 1) {
    size_t j1 = 0;
    for (size_t i = 0; i < m; i++, root_index++) {
      size_t j2 = j1 + t;
      const uint64_t W_op = inv_root_of_unity_powers[root_index];
      const uint64_t W_op_precon = precon_inv_root_of_unity_powers[root_index];

      HEXL_VLOG(4, "m = " << i << ", i = " << i);
      HEXL_VLOG(4, "j1 = " << j1 << ", j2 = " << j2);

      uint64_t* X = operand + j1;
      uint64_t* Y = X + t;

      uint64_t tx;
      uint64_t ty;

      HEXL_LOOP_UNROLL_4
      for (size_t j = j1; j < j2; j++) {
        HEXL_VLOG(4, "Loaded *X " << *X);
        HEXL_VLOG(4, "Loaded *Y " << *Y);
        // The Harvey butterfly: assume X, Y in [0, 2q), and return X', Y'
        // in [0, 2q). X', Y' = X + Y (mod q), W(X - Y) (mod q).
        tx = *X + *Y;
        ty = *X + twice_mod - *Y;

        *X++ = (tx >= twice_mod) ? (tx - twice_mod) : tx;
        *Y++ = MultiplyUIntModLazy<64>(ty, W_op, W_op_precon, modulus);
      }
      j1 += (t << 1);
    }
    t <<= 1;
  }

  const uint64_t W_op = inv_root_of_unity_powers[root_index];
  const uint64_t inv_n = InverseUIntMod(n, modulus);
  const uint64_t inv_n_w = MultiplyUIntMod(inv_n, W_op, modulus);

  uint64_t* X = operand;
  uint64_t* Y = X + (n >> 1);
  uint64_t tx;
  uint64_t ty;

  for (size_t j = (n >> 1); j < n; j++) {
    tx = *X + *Y;
    if (tx >= twice_mod) {
      tx -= twice_mod;
    }
    ty = *X + twice_mod - *Y;
    *X++ = MultiplyUIntModLazy<64>(tx, inv_n, modulus);
    *Y++ = MultiplyUIntModLazy<64>(ty, inv_n_w, modulus);
  }

  if (output_mod_factor == 1) {
    // Reduce from [0, 2q) to [0,q)
    for (size_t i = 0; i < n; ++i) {
      if (operand[i] >= modulus) {
        operand[i] -= modulus;
      }
      HEXL_CHECK(operand[i] < modulus, "Incorrect modulus reduction in InvNTT"
                                           << operand[i] << " >= " << modulus);
    }
  }
}

bool CheckNTTArguments(uint64_t degree, uint64_t modulus) {
  // Avoid unused parameter warnings
  (void)degree;
  (void)modulus;
  HEXL_CHECK(IsPowerOfTwo(degree),
             "degree " << degree << " is not a power of 2");
  HEXL_CHECK(degree <= (1 << NTT::NTTImpl::s_max_degree_bits),
             "degree should be less than 2^" << NTT::NTTImpl::s_max_degree_bits
                                             << " got " << degree);

  HEXL_CHECK(modulus % (2 * degree) == 1, "modulus mod 2n != 1");
  return true;
}

}  // namespace hexl
}  // namespace intel
