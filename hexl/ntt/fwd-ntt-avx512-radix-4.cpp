// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <functional>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/ntt-avx512-util.hpp"
#include "ntt/ntt-default.hpp"
#include "ntt/ntt-internal.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

/// @brief The Harvey butterfly: assume \p X, \p Y in [0, 4q), and return X', Y'
/// in [0, 4q) such that X', Y' = X + WY, X - WY (mod q).
/// @param[in,out] X Input representing 8 64-bit signed integers in SIMD form
/// @param[in,out] Y Input representing 8 64-bit signed integers in SIMD form
/// @param[in] W Root of unity represented as 8 64-bit signed integers in
/// SIMD form
/// @param[in] W_precon Preconditioned \p W for BitShift-bit Barrett
/// reduction
/// @param[in] neg_modulus Negative modulus, i.e. (-q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @param InputLessThanMod If true, assumes \p X, \p Y < \p q. Otherwise,
/// assumes \p X, \p Y < 4*\p q
/// @details See Algorithm 4 of https://arxiv.org/pdf/1205.2926.pdf
template <int BitShift, bool InputLessThanMod>
void FwdButterflyRadix2AVX512(__m512i* X, __m512i* Y, __m512i W,
                              __m512i W_precon, __m512i neg_modulus,
                              __m512i twice_modulus) {
  if (!InputLessThanMod) {
    *X = _mm512_hexl_small_mod_epu64(*X, twice_modulus);
  }

  __m512i T;
  if (BitShift == 32) {
    __m512i Q = _mm512_hexl_mullo_epi<64>(W_precon, *Y);
    Q = _mm512_srli_epi64(Q, 32);
    __m512i W_Y = _mm512_hexl_mullo_epi<64>(W, *Y);
    T = _mm512_hexl_mullo_add_lo_epi<64>(W_Y, Q, neg_modulus);
  } else if (BitShift == 52) {
    __m512i Q = _mm512_hexl_mulhi_epi<BitShift>(W_precon, *Y);
    __m512i W_Y = _mm512_hexl_mullo_epi<BitShift>(W, *Y);
    T = _mm512_hexl_mullo_add_lo_epi<BitShift>(W_Y, Q, neg_modulus);
  } else if (BitShift == 64) {
    // Perform approximate computation of Q, as described in page 7 of
    // https://arxiv.org/pdf/2003.04510.pdf
    __m512i Q = _mm512_hexl_mulhi_approx_epi<BitShift>(W_precon, *Y);
    __m512i W_Y = _mm512_hexl_mullo_epi<BitShift>(W, *Y);
    // Compute T in range [0, 4q)
    T = _mm512_hexl_mullo_add_lo_epi<BitShift>(W_Y, Q, neg_modulus);
    // Reduce T to range [0, 2q)
    T = _mm512_hexl_small_mod_epu64<2>(T, twice_modulus);
  } else {
    HEXL_CHECK(false, "Invalid BitShift " << BitShift);
  }

  __m512i twice_mod_minus_T = _mm512_sub_epi64(twice_modulus, T);
  *Y = _mm512_add_epi64(*X, twice_mod_minus_T);
  *X = _mm512_add_epi64(*X, T);
}

// Assume X0, X1, X2, X3 in [0, 4q) and return X0, X1, X2, X3 in [0, 4q)
template <int BitShift = 64>
inline void FwdButterflyRadix4AVX512(__m512i* X_op0, __m512i* X_op1,
                                     __m512i* X_op2, __m512i* X_op3, __m512i W1,
                                     __m512i W1_precon, __m512i W2,
                                     __m512i W2_precon, __m512i W3,
                                     __m512i W3_precon, __m512i modulus,
                                     __m512i twice_modulus) {
  // uint64_t* X_r0, uint64_t* X_r1, uint64_t* X_r2, uint64_t* X_r3,
  // const uint64_t* X_op0, const uint64_t* X_op1, const uint64_t* X_op2,
  // const uint64_t* X_op3, uint64_t W1, uint64_t W1_precon, uint64_t W2,
  // uint64_t W2_precon, uint64_t W3, uint64_t W3_precon, uint64_t modulus,
  // uint64_t twice_modulus, uint64_t four_times_modulus) {
  HEXL_VLOG(3, "FwdButterflyRadix4AVX512 BitShift " << BitShift);
  // HEXL_UNUSED(four_times_modulus);

  FwdButterflyRadix2AVX512<BitShift, false>(X_op0, X_op2, W1, W1_precon,
                                            modulus, twice_modulus);
  FwdButterflyRadix2AVX512<BitShift, false>(X_op1, X_op3, W1, W1_precon,
                                            modulus, twice_modulus);
  FwdButterflyRadix2AVX512<BitShift, false>(X_op0, X_op1, W2, W2_precon,
                                            modulus, twice_modulus);
  FwdButterflyRadix2AVX512<BitShift, false>(X_op2, X_op3, W3, W3_precon,
                                            modulus, twice_modulus);
  // Alternate implementation
  // // Returns Xs in [0, 6q)
  // FwdButterflyRadix4Lazy(X0, X2, W1, W1_precon, modulus, twice_modulus);
  // FwdButterflyRadix4Lazy(X1, X3, W1, W1_precon, modulus, twice_modulus);

  // // Returns Xs in [0, 8q)
  // FwdButterflyRadix4Lazy(X0, X1, W2, W2_precon, modulus, twice_modulus);
  // FwdButterflyRadix4Lazy(X2, X3, W3, W3_precon, modulus, twice_modulus);

  // // Reduce Xs to [0, 4q)
  // *X0 = ReduceMod<2>(*X0, four_times_modulus);
  // *X1 = ReduceMod<2>(*X1, four_times_modulus);
  // *X2 = ReduceMod<2>(*X2, four_times_modulus);
  // *X3 = ReduceMod<2>(*X3, four_times_modulus);
}

// Out-of-place implementation
template <int BitShift, bool InputLessThanMod>
void FwdT8Radix2(uint64_t* result, const uint64_t* operand,
                 __m512i v_neg_modulus, __m512i v_twice_mod, uint64_t t,
                 uint64_t m, const uint64_t* W, const uint64_t* W_precon) {
  HEXL_CHECK(t % 8 == 0, "Invalid t " << t << "; t must be divisible by 8")
  size_t j1 = 0;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < m; i++) {
    // Referencing operand
    const uint64_t* X_op = operand + j1;
    const uint64_t* Y_op = X_op + t;

    const __m512i* v_X_op_pt = reinterpret_cast<const __m512i*>(X_op);
    const __m512i* v_Y_op_pt = reinterpret_cast<const __m512i*>(Y_op);

    // Referencing result
    uint64_t* X_r = result + j1;
    uint64_t* Y_r = X_r + t;

    __m512i* v_X_r_pt = reinterpret_cast<__m512i*>(X_r);
    __m512i* v_Y_r_pt = reinterpret_cast<__m512i*>(Y_r);

    // Weights and weights' preconditions
    __m512i v_W = _mm512_set1_epi64(static_cast<int64_t>(*W++));
    __m512i v_W_precon = _mm512_set1_epi64(static_cast<int64_t>(*W_precon++));

    // assume 8 | t
    for (size_t j = t / 8; j > 0; --j) {
      __m512i v_X = _mm512_loadu_si512(v_X_op_pt);
      __m512i v_Y = _mm512_loadu_si512(v_Y_op_pt);

      FwdButterflyRadix2AVX512<BitShift, InputLessThanMod>(
          &v_X, &v_Y, v_W, v_W_precon, v_neg_modulus, v_twice_mod);

      _mm512_storeu_si512(v_X_r_pt++, v_X);
      _mm512_storeu_si512(v_Y_r_pt++, v_Y);

      // Increase operand pointers as well
      v_X_op_pt++;
      v_Y_op_pt++;
    }
    j1 += (t << 1);
  }
}

#ifdef HEXL_HAS_AVX512IFMA
template void ForwardTransformToBitReverseAVX512Radix4<NTT::s_ifma_shift_bits>(
    uint64_t* result, const uint64_t* operand, uint64_t degree, uint64_t mod,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);
#endif

#ifdef HEXL_HAS_AVX512DQ
template void ForwardTransformToBitReverseAVX512Radix4<32>(
    uint64_t* result, const uint64_t* operand, uint64_t degree, uint64_t mod,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);

template void
ForwardTransformToBitReverseAVX512Radix4<NTT::s_default_shift_bits>(
    uint64_t* result, const uint64_t* operand, uint64_t degree, uint64_t mod,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);
#endif

#ifdef HEXL_HAS_AVX512DQ

template <int BitShift>
void FwdT1(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W, const uint64_t* W_precon) {
  const __m512i* v_W_pt = reinterpret_cast<const __m512i*>(W);
  const __m512i* v_W_precon_pt = reinterpret_cast<const __m512i*>(W_precon);
  size_t j1 = 0;

  // 8 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_8
  for (size_t i = m / 8; i > 0; --i) {
    uint64_t* X = operand + j1;
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadFwdInterleavedT1(X, &v_X, &v_Y);
    __m512i v_W = _mm512_loadu_si512(v_W_pt++);
    __m512i v_W_precon = _mm512_loadu_si512(v_W_precon_pt++);

    FwdButterflyRadix2<BitShift, false>(&v_X, &v_Y, v_W, v_W_precon,
                                        v_neg_modulus, v_twice_mod);
    WriteFwdInterleavedT1(v_X, v_Y, v_X_pt);

    j1 += 16;
  }
}

template <int BitShift>
void FwdT2(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W, const uint64_t* W_precon) {
  const __m512i* v_W_pt = reinterpret_cast<const __m512i*>(W);
  const __m512i* v_W_precon_pt = reinterpret_cast<const __m512i*>(W_precon);

  size_t j1 = 0;
  // 4 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = m / 4; i > 0; --i) {
    uint64_t* X = operand + j1;
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadFwdInterleavedT2(X, &v_X, &v_Y);

    __m512i v_W = _mm512_loadu_si512(v_W_pt++);
    __m512i v_W_precon = _mm512_loadu_si512(v_W_precon_pt++);

    HEXL_CHECK(ExtractValues(v_W)[0] == ExtractValues(v_W)[1],
               "bad v_W " << ExtractValues(v_W));
    HEXL_CHECK(ExtractValues(v_W_precon)[0] == ExtractValues(v_W_precon)[1],
               "bad v_W_precon " << ExtractValues(v_W_precon));
    FwdButterflyRadix2<BitShift, false>(&v_X, &v_Y, v_W, v_W_precon,
                                        v_neg_modulus, v_twice_mod);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_X_pt, v_Y);

    j1 += 16;
  }
}

template <int BitShift>
void FwdT4(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W, const uint64_t* W_precon) {
  size_t j1 = 0;
  const __m512i* v_W_pt = reinterpret_cast<const __m512i*>(W);
  const __m512i* v_W_precon_pt = reinterpret_cast<const __m512i*>(W_precon);

  // 2 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = m / 2; i > 0; --i) {
    uint64_t* X = operand + j1;
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadFwdInterleavedT4(X, &v_X, &v_Y);

    __m512i v_W = _mm512_loadu_si512(v_W_pt++);
    __m512i v_W_precon = _mm512_loadu_si512(v_W_precon_pt++);
    FwdButterflyRadix2<BitShift, false>(&v_X, &v_Y, v_W, v_W_precon,
                                        v_neg_modulus, v_twice_mod);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_X_pt, v_Y);

    j1 += 16;
  }
}

// Out-of-place implementation
template <int BitShift, bool InputLessThanMod>
void FwdT8(uint64_t* result, const uint64_t* operand, __m512i v_neg_modulus,
           __m512i v_twice_mod, uint64_t t, uint64_t m, const uint64_t* W,
           const uint64_t* W_precon) {
  size_t j1 = 0;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < m; i++) {
    // Referencing operand
    const uint64_t* X_op = operand + j1;
    const uint64_t* Y_op = X_op + t;

    const __m512i* v_X_op_pt = reinterpret_cast<const __m512i*>(X_op);
    const __m512i* v_Y_op_pt = reinterpret_cast<const __m512i*>(Y_op);

    // Referencing result
    uint64_t* X_r = result + j1;
    uint64_t* Y_r = X_r + t;

    __m512i* v_X_r_pt = reinterpret_cast<__m512i*>(X_r);
    __m512i* v_Y_r_pt = reinterpret_cast<__m512i*>(Y_r);

    // Weights and weights' preconditions
    __m512i v_W = _mm512_set1_epi64(static_cast<int64_t>(*W++));
    __m512i v_W_precon = _mm512_set1_epi64(static_cast<int64_t>(*W_precon++));

    // assume 8 | t
    for (size_t j = t / 8; j > 0; --j) {
      __m512i v_X = _mm512_loadu_si512(v_X_op_pt);
      __m512i v_Y = _mm512_loadu_si512(v_Y_op_pt);

      FwdButterflyRadix2<BitShift, InputLessThanMod>(
          &v_X, &v_Y, v_W, v_W_precon, v_neg_modulus, v_twice_mod);

      _mm512_storeu_si512(v_X_r_pt++, v_X);
      _mm512_storeu_si512(v_Y_r_pt++, v_Y);

      // Increase operand pointers as well
      v_X_op_pt++;
      v_Y_op_pt++;
    }
    j1 += (t << 1);
  }
}

template <int BitShift>
void ForwardTransformToBitReverseAVX512Radix4(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half) {
  HEXL_VLOG(4, "ForwardTransformToBitReverseAVX512Radix4");
  HEXL_CHECK(NTT::CheckArguments(n, modulus), "");
  HEXL_CHECK(modulus < NTT::s_max_fwd_modulus(BitShift),
             "modulus " << modulus << " too large for BitShift " << BitShift
                        << " => maximum value "
                        << NTT::s_max_fwd_modulus(BitShift));
  HEXL_CHECK_BOUNDS(precon_root_of_unity_powers, n, MaximumValue(BitShift),
                    "precon_root_of_unity_powers too large");
  HEXL_CHECK_BOUNDS(operand, n, MaximumValue(BitShift), "operand too large");
  // Skip input bound checking for recursive steps
  HEXL_CHECK_BOUNDS(operand, (recursion_depth == 0) ? n : 0,
                    input_mod_factor * modulus,
                    "operand larger than input_mod_factor * modulus ("
                        << input_mod_factor << " * " << modulus << ")");
  HEXL_CHECK(n >= 16,
             "Don't support small transforms. Need n >= 16, got n = " << n);
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "input_mod_factor must be 1, 2, or 4; got " << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
             "output_mod_factor must be 1 or 4; got " << output_mod_factor);

  uint64_t twice_mod = modulus << 1;

  HEXL_CHECK(NTT::CheckArguments(n, modulus), "");
  HEXL_CHECK_BOUNDS(operand, n, modulus * input_mod_factor,
                    "operand exceeds bound " << modulus * input_mod_factor);
  HEXL_CHECK(root_of_unity_powers != nullptr,
             "root_of_unity_powers == nullptr");
  HEXL_CHECK(precon_root_of_unity_powers != nullptr,
             "precon_root_of_unity_powers == nullptr");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4,
      "input_mod_factor must be 1, 2, or 4; got " << input_mod_factor);
  HEXL_UNUSED(input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 4,
             "output_mod_factor must be 1 or 4; got " << output_mod_factor);

  HEXL_VLOG(3, "modulus " << modulus);
  HEXL_VLOG(3, "n " << n);

  HEXL_VLOG(3, "operand " << std::vector<uint64_t>(operand, operand + n));

  HEXL_VLOG(3, "root_of_unity_powers " << std::vector<uint64_t>(
                   root_of_unity_powers, root_of_unity_powers + n));

  bool is_power_of_4 = IsPowerOfFour(n);

  uint64_t twice_modulus = modulus << 1;
  uint64_t four_times_modulus = modulus << 2;

  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_neg_modulus = _mm512_set1_epi64(-static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(twice_modulus));

  // Radix-2 step for non-powers of 4
  if (!is_power_of_4) {
    HEXL_VLOG(3, "Radix 2 step");
    size_t t = (n >> 1);
    HEXL_VLOG(3, "radix 2 t " << t);

    uint64_t* X_r = result;
    const uint64_t* X_op = operand;
    const uint64_t* W = &root_of_unity_powers[1];
    const uint64_t* W_precon = &precon_root_of_unity_powers[1];

    FwdT8Radix2<BitShift, false>(X_r, X_op, v_neg_modulus, v_twice_mod, t, 1, W,
                                 W_precon);

    //  uint64_t* Y_r = X_r + t;
    //  const uint64_t* Y_op = X_op + t;
    // HEXL_LOOP_UNROLL_8
    // for (size_t j = 0; j < t; j++) {
    //   const uint64_t W = root_of_unity_powers[1];
    //   const uint64_t W_precon = precon_root_of_unity_powers[1];
    //   FwdButterflyRadix2<BitShift>(X_r++, Y_r++, X_op++, Y_op++, W, W_precon,
    //                                modulus, twice_modulus);
    // }
    // Data in [0, 4q)
    HEXL_VLOG(3, "after radix 2 outputs "
                     << std::vector<uint64_t>(result, result + n));
  }

  uint64_t m_start = 2;
  size_t t = n >> 3;
  if (is_power_of_4) {
    t = n >> 2;

    uint64_t* X_r0 = result;
    uint64_t* X_r1 = X_r0 + t;
    uint64_t* X_r2 = X_r0 + 2 * t;
    uint64_t* X_r3 = X_r0 + 3 * t;
    const uint64_t* X_op0 = operand;
    const uint64_t* X_op1 = operand + t;
    const uint64_t* X_op2 = operand + 2 * t;
    const uint64_t* X_op3 = operand + 3 * t;

    uint64_t W1_ind = 1;
    uint64_t W2_ind = 2 * W1_ind;
    uint64_t W3_ind = 2 * W1_ind + 1;

    const uint64_t W1 = root_of_unity_powers[W1_ind];
    const uint64_t W2 = root_of_unity_powers[W2_ind];
    const uint64_t W3 = root_of_unity_powers[W3_ind];

    const uint64_t W1_precon = precon_root_of_unity_powers[W1_ind];
    const uint64_t W2_precon = precon_root_of_unity_powers[W2_ind];
    const uint64_t W3_precon = precon_root_of_unity_powers[W3_ind];

    switch (t) {
      case 4: {
        FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                     X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                     W2, W2_precon, W3, W3_precon, modulus,
                                     twice_modulus, four_times_modulus);
        FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                     X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                     W2, W2_precon, W3, W3_precon, modulus,
                                     twice_modulus, four_times_modulus);
        FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                     X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                     W2, W2_precon, W3, W3_precon, modulus,
                                     twice_modulus, four_times_modulus);
        FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                     X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                     W2, W2_precon, W3, W3_precon, modulus,
                                     twice_modulus, four_times_modulus);
        break;
      }
      case 1: {
        FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                     X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                     W2, W2_precon, W3, W3_precon, modulus,
                                     twice_modulus, four_times_modulus);
        break;
      }
      default: {
        for (size_t j = 0; j < t; j += 16) {
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
        }
      }
    }
    t >>= 2;
    m_start = 4;
  }

  // uint64_t m_start = is_power_of_4 ? 1 : 2;
  // size_t t = (n >> m_start) >> 1;

  for (size_t m = m_start; m < n; m <<= 2) {
    HEXL_VLOG(3, "m " << m);
    HEXL_VLOG(3, "t " << t);

    size_t X0_offset = 0;

    switch (t) {
      case 4: {
        HEXL_LOOP_UNROLL_8
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            X0_offset += 4 * t;
          }
          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = X_r0;
          const uint64_t* X_op1 = X_r1;
          const uint64_t* X_op2 = X_r2;
          const uint64_t* X_op3 = X_r3;

          uint64_t W1_ind = m + i;
          uint64_t W2_ind = 2 * W1_ind;
          uint64_t W3_ind = 2 * W1_ind + 1;

          const uint64_t W1 = root_of_unity_powers[W1_ind];
          const uint64_t W2 = root_of_unity_powers[W2_ind];
          const uint64_t W3 = root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_root_of_unity_powers[W3_ind];

          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
        }
        break;
      }
      case 1: {
        HEXL_LOOP_UNROLL_8
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            X0_offset += 4 * t;
          }
          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = X_r0;
          const uint64_t* X_op1 = X_r1;
          const uint64_t* X_op2 = X_r2;
          const uint64_t* X_op3 = X_r3;

          uint64_t W1_ind = m + i;
          uint64_t W2_ind = 2 * W1_ind;
          uint64_t W3_ind = 2 * W1_ind + 1;

          const uint64_t W1 = root_of_unity_powers[W1_ind];
          const uint64_t W2 = root_of_unity_powers[W2_ind];
          const uint64_t W3 = root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_root_of_unity_powers[W3_ind];

          FwdButterflyRadix4<BitShift>(X_r0++, X_r1++, X_r2++, X_r3++, X_op0++,
                                       X_op1++, X_op2++, X_op3++, W1, W1_precon,
                                       W2, W2_precon, W3, W3_precon, modulus,
                                       twice_modulus, four_times_modulus);
        }
        break;
      }
      default: {
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            X0_offset += 4 * t;
          }

          uint64_t* X_r0 = result + X0_offset;
          uint64_t* X_r1 = X_r0 + t;
          uint64_t* X_r2 = X_r0 + 2 * t;
          uint64_t* X_r3 = X_r0 + 3 * t;
          const uint64_t* X_op0 = X_r0;
          const uint64_t* X_op1 = X_r1;
          const uint64_t* X_op2 = X_r2;
          const uint64_t* X_op3 = X_r3;

          uint64_t W1_ind = m + i;
          uint64_t W2_ind = 2 * W1_ind;
          uint64_t W3_ind = 2 * W1_ind + 1;

          const uint64_t W1 = root_of_unity_powers[W1_ind];
          const uint64_t W2 = root_of_unity_powers[W2_ind];
          const uint64_t W3 = root_of_unity_powers[W3_ind];

          const uint64_t W1_precon = precon_root_of_unity_powers[W1_ind];
          const uint64_t W2_precon = precon_root_of_unity_powers[W2_ind];
          const uint64_t W3_precon = precon_root_of_unity_powers[W3_ind];

          __m512i v_W1 = _mm512_set1_epi64(static_cast<int64_t>(W1));
          __m512i v_W1_precon =
              _mm512_set1_epi64(static_cast<int64_t>(W1_precon));
          __m512i v_W2 = _mm512_set1_epi64(static_cast<int64_t>(W2));
          __m512i v_W2_precon =
              _mm512_set1_epi64(static_cast<int64_t>(W2_precon));
          __m512i v_W3 = _mm512_set1_epi64(static_cast<int64_t>(W3));
          __m512i v_W3_precon =
              _mm512_set1_epi64(static_cast<int64_t>(W3_precon));

          __m512i* v_X_op0_pt =
              const_cast<__m512i*>(reinterpret_cast<const __m512i*>(X_op0));
          __m512i* v_X_op1_pt =
              const_cast<__m512i*>(reinterpret_cast<const __m512i*>(X_op1));
          __m512i* v_X_op2_pt =
              const_cast<__m512i*>(reinterpret_cast<const __m512i*>(X_op2));
          __m512i* v_X_op3_pt =
              const_cast<__m512i*>(reinterpret_cast<const __m512i*>(X_op3));

          HEXL_CHECK(t % 8 == 0, "t % 8 must be 0");

          for (size_t j = 0; j < t / 8; j++) {
            __m512i v_Xop0 = _mm512_loadu_si512(v_X_op0_pt);
            __m512i v_Xop1 = _mm512_loadu_si512(v_X_op1_pt);
            __m512i v_Xop2 = _mm512_loadu_si512(v_X_op2_pt);
            __m512i v_Xop3 = _mm512_loadu_si512(v_X_op3_pt);

            FwdButterflyRadix4AVX512<BitShift>(
                &v_Xop0, &v_Xop1, &v_Xop2, &v_Xop3, v_W1, v_W1_precon, v_W2,
                v_W2_precon, v_W3, v_W3_precon, v_neg_modulus, v_twice_mod);

            _mm512_storeu_si512(v_X_op0_pt++, v_Xop0);
            _mm512_storeu_si512(v_X_op1_pt++, v_Xop1);
            _mm512_storeu_si512(v_X_op2_pt++, v_Xop2);
            _mm512_storeu_si512(v_X_op3_pt++, v_Xop3);
          }
        }
      }
    }
    t >>= 2;
  }

  if (output_mod_factor == 1) {
    for (size_t i = 0; i < n; ++i) {
      if (result[i] >= twice_modulus) {
        result[i] -= twice_modulus;
      }
      if (result[i] >= modulus) {
        result[i] -= modulus;
      }
      HEXL_CHECK(result[i] < modulus, "Incorrect modulus reduction in NTT "
                                          << result[i] << " >= " << modulus);
    }
  }

  HEXL_VLOG(3, "outputs " << std::vector<uint64_t>(result, result + n));

  // HEXL_VLOG(5, "root_of_unity_powers " << std::vector<uint64_t>(
  //                  root_of_unity_powers, root_of_unity_powers + n))
  // HEXL_VLOG(5,
  //           "precon_root_of_unity_powers " << std::vector<uint64_t>(
  //               precon_root_of_unity_powers, precon_root_of_unity_powers +
  //               n));
  // HEXL_VLOG(5, "operand " << std::vector<uint64_t>(operand, operand + n));

  // static const size_t base_ntt_size = 1024;

  // if (n <= base_ntt_size) {  // Perform breadth-first NTT
  //   size_t t = (n >> 1);
  //   size_t m = 1;
  //   size_t W_idx = (m << recursion_depth) + (recursion_half * m);

  //   // Copy for out-of-place in case m is <= base_ntt_size from start
  //   if (result != operand) {
  //     std::memcpy(result, operand, n * sizeof(uint64_t));
  //   }

  //   // First iteration assumes input in [0,p)
  //   if (m < (n >> 3)) {
  //     const uint64_t* W = &root_of_unity_powers[W_idx];
  //     const uint64_t* W_precon = &precon_root_of_unity_powers[W_idx];

  //     if ((input_mod_factor <= 2) && (recursion_depth == 0)) {
  //       FwdT8<BitShift, true>(result, result, v_neg_modulus, v_twice_mod, t,
  //       m,
  //                             W, W_precon);
  //     } else {
  //       FwdT8<BitShift, false>(result, result, v_neg_modulus, v_twice_mod, t,
  //       m,
  //                              W, W_precon);
  //     }

  //     t >>= 1;
  //     m <<= 1;
  //     W_idx <<= 1;
  //   }
  //   for (; m < (n >> 3); m <<= 1) {
  //     const uint64_t* W = &root_of_unity_powers[W_idx];
  //     const uint64_t* W_precon = &precon_root_of_unity_powers[W_idx];
  //     FwdT8<BitShift, false>(result, result, v_neg_modulus, v_twice_mod, t,
  //     m,
  //                            W, W_precon);
  //     t >>= 1;
  //     W_idx <<= 1;
  //   }

  //   // Do T=4, T=2, T=1 separately
  //   {
  //     // Correction step needed due to extra copies of roots of unity in the
  //     // AVX512 vectors loaded for FwdT2 and FwdT4
  //     auto compute_new_W_idx = [&](size_t idx) {
  //       // Originally, from root of unity vector index to loop:
  //       // [0, N/8) => FwdT8
  //       // [N/8, N/4) => FwdT4
  //       // [N/4, N/2) => FwdT2
  //       // [N/2, N) => FwdT1
  //       // The new mapping from AVX512 root of unity vector index to loop:
  //       // [0, N/8) => FwdT8
  //       // [N/8, 5N/8) => FwdT4
  //       // [5N/8, 9N/8) => FwdT2
  //       // [9N/8, 13N/8) => FwdT1
  //       size_t N = n << recursion_depth;

  //       // FwdT8 range
  //       if (idx <= N / 8) {
  //         return idx;
  //       }
  //       // FwdT4 range
  //       if (idx <= N / 4) {
  //         return (idx - N / 8) * 4 + (N / 8);
  //       }
  //       // FwdT2 range
  //       if (idx <= N / 2) {
  //         return (idx - N / 4) * 2 + (5 * N / 8);
  //       }
  //       // FwdT1 range
  //       return idx + (5 * N / 8);
  //     };

  //     size_t new_W_idx = compute_new_W_idx(W_idx);
  //     const uint64_t* W = &root_of_unity_powers[new_W_idx];
  //     const uint64_t* W_precon = &precon_root_of_unity_powers[new_W_idx];
  //     FwdT4<BitShift>(result, v_neg_modulus, v_twice_mod, m, W, W_precon);

  //     m <<= 1;
  //     W_idx <<= 1;
  //     new_W_idx = compute_new_W_idx(W_idx);
  //     W = &root_of_unity_powers[new_W_idx];
  //     W_precon = &precon_root_of_unity_powers[new_W_idx];
  //     FwdT2<BitShift>(result, v_neg_modulus, v_twice_mod, m, W, W_precon);

  //     m <<= 1;
  //     W_idx <<= 1;
  //     new_W_idx = compute_new_W_idx(W_idx);
  //     W = &root_of_unity_powers[new_W_idx];
  //     W_precon = &precon_root_of_unity_powers[new_W_idx];
  //     FwdT1<BitShift>(result, v_neg_modulus, v_twice_mod, m, W, W_precon);
  //   }

  //   if (output_mod_factor == 1) {
  //     // n power of two at least 8 => n divisible by 8
  //     HEXL_CHECK(n % 8 == 0, "n " << n << " not a power of 2");
  //     __m512i* v_X_pt = reinterpret_cast<__m512i*>(result);
  //     for (size_t i = 0; i < n; i += 8) {
  //       __m512i v_X = _mm512_loadu_si512(v_X_pt);

  //       // Reduce from [0, 4q) to [0, q)
  //       v_X = _mm512_hexl_small_mod_epu64(v_X, v_twice_mod);
  //       v_X = _mm512_hexl_small_mod_epu64(v_X, v_modulus);

  //       HEXL_CHECK_BOUNDS(ExtractValues(v_X).data(), 8, modulus,
  //                         "v_X exceeds bound " << modulus);

  //       _mm512_storeu_si512(v_X_pt, v_X);

  //       ++v_X_pt;
  //     }
  //   }
  // } else {
  //   // Perform depth-first NTT via recursive call
  //   size_t t = (n >> 1);
  //   size_t W_idx = (1ULL << recursion_depth) + recursion_half;
  //   const uint64_t* W = &root_of_unity_powers[W_idx];
  //   const uint64_t* W_precon = &precon_root_of_unity_powers[W_idx];

  //   FwdT8<BitShift, false>(result, operand, v_neg_modulus, v_twice_mod, t, 1,
  //   W,
  //                          W_precon);

  //   ForwardTransformToBitReverseAVX512<BitShift>(
  //       result, result, n / 2, modulus, root_of_unity_powers,
  //       precon_root_of_unity_powers, input_mod_factor, output_mod_factor,
  //       recursion_depth + 1, recursion_half * 2);

  //   ForwardTransformToBitReverseAVX512<BitShift>(
  //       &result[n / 2], &result[n / 2], n / 2, modulus, root_of_unity_powers,
  //       precon_root_of_unity_powers, input_mod_factor, output_mod_factor,
  //       recursion_depth + 1, recursion_half * 2 + 1);
  // }
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
