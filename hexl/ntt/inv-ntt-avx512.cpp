// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ntt/inv-ntt-avx512.hpp"

#include <immintrin.h>

#include <cstring>
#include <functional>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "ntt/ntt-avx512-util.hpp"
#include "ntt/ntt-internal.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512IFMA
template void InverseTransformFromBitReverseAVX512<NTT::s_ifma_shift_bits>(
    uint64_t* result, const uint64_t* operand, uint64_t degree,
    uint64_t modulus, const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);
#endif

#ifdef HEXL_HAS_AVX512DQ
template void InverseTransformFromBitReverseAVX512<32>(
    uint64_t* result, const uint64_t* operand, uint64_t degree,
    uint64_t modulus, const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);

template void InverseTransformFromBitReverseAVX512<NTT::s_default_shift_bits>(
    uint64_t* result, const uint64_t* operand, uint64_t degree,
    uint64_t modulus, const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);
#endif

#ifdef HEXL_HAS_AVX512DQ

/// @brief The Harvey butterfly: assume X, Y in [0, 2q), and return X', Y' in
/// [0, 2q). such that X', Y' = X + Y (mod q), W(X - Y) (mod q).
/// @param[in,out] X Input representing 8 64-bit signed integers in SIMD form
/// @param[in,out] Y Input representing 8 64-bit signed integers in SIMD form
/// @param[in] W Root of unity representing 8 64-bit signed integers in SIMD
/// form
/// @param[in] W_precon Preconditioned \p W for BitShift-bit Barrett
/// reduction
/// @param[in] neg_modulus Negative modulus, i.e. (-q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @param InputLessThanMod If true, assumes \p X, \p Y < \p q. Otherwise,
/// assumes \p X, \p Y < 2*\p q
/// @details See Algorithm 3 of https://arxiv.org/pdf/1205.2926.pdf
template <int BitShift, bool InputLessThanMod>
inline void InvButterfly(__m512i* X, __m512i* Y, __m512i W, __m512i W_precon,
                         __m512i neg_modulus, __m512i twice_modulus) {
  __m512i Y_minus_2q = _mm512_sub_epi64(*Y, twice_modulus);
  __m512i T = _mm512_sub_epi64(*X, Y_minus_2q);

  if (InputLessThanMod) {
    // No need for modulus reduction, since inputs are in [0, q)
    *X = _mm512_add_epi64(*X, *Y);
  } else {
    *X = _mm512_add_epi64(*X, Y_minus_2q);
    __mmask8 sign_bits = _mm512_movepi64_mask(*X);
    *X = _mm512_mask_add_epi64(*X, sign_bits, *X, twice_modulus);
  }

  if (BitShift == 32) {
    __m512i Q = _mm512_hexl_mullo_epi<64>(W_precon, T);
    Q = _mm512_srli_epi64(Q, 32);
    __m512i Q_p = _mm512_hexl_mullo_epi<64>(Q, neg_modulus);
    *Y = _mm512_hexl_mullo_add_lo_epi<64>(Q_p, W, T);
  } else if (BitShift == 52) {
    __m512i Q = _mm512_hexl_mulhi_epi<BitShift>(W_precon, T);
    __m512i Q_p = _mm512_hexl_mullo_epi<BitShift>(Q, neg_modulus);
    *Y = _mm512_hexl_mullo_add_lo_epi<BitShift>(Q_p, W, T);
  } else if (BitShift == 64) {
    // Perform approximate computation of Q, as described in page 7 of
    // https://arxiv.org/pdf/2003.04510.pdf
    __m512i Q = _mm512_hexl_mulhi_approx_epi<BitShift>(W_precon, T);
    __m512i Q_p = _mm512_hexl_mullo_epi<BitShift>(Q, neg_modulus);
    // Compute Y in range [0, 4q)
    *Y = _mm512_hexl_mullo_add_lo_epi<BitShift>(Q_p, W, T);
    // Reduce Y to range [0, 2q)
    *Y = _mm512_hexl_small_mod_epu64<2>(*Y, twice_modulus);
  } else {
    HEXL_CHECK(false, "Invalid BitShift " << BitShift);
  }
}

template <int BitShift, bool InputLessThanMod>
void InvT1(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
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
    LoadInvInterleavedT1(X, &v_X, &v_Y);

    __m512i v_W = _mm512_loadu_si512(v_W_pt++);
    __m512i v_W_precon = _mm512_loadu_si512(v_W_precon_pt++);

    InvButterfly<BitShift, InputLessThanMod>(&v_X, &v_Y, v_W, v_W_precon,
                                             v_neg_modulus, v_twice_mod);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_X_pt, v_Y);

    j1 += 16;
  }
}

template <int BitShift>
void InvT2(uint64_t* X, __m512i v_neg_modulus, __m512i v_twice_mod, uint64_t m,
           const uint64_t* W, const uint64_t* W_precon) {
  // 4 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = m / 4; i > 0; --i) {
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadInvInterleavedT2(X, &v_X, &v_Y);

    __m512i v_W = LoadWOpT2(static_cast<const void*>(W));
    __m512i v_W_precon = LoadWOpT2(static_cast<const void*>(W_precon));

    InvButterfly<BitShift, false>(&v_X, &v_Y, v_W, v_W_precon, v_neg_modulus,
                                  v_twice_mod);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_X_pt, v_Y);
    X += 16;

    W += 4;
    W_precon += 4;
  }
}

template <int BitShift>
void InvT4(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W, const uint64_t* W_precon) {
  uint64_t* X = operand;

  // 2 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = m / 2; i > 0; --i) {
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadInvInterleavedT4(X, &v_X, &v_Y);

    __m512i v_W = LoadWOpT4(static_cast<const void*>(W));
    __m512i v_W_precon = LoadWOpT4(static_cast<const void*>(W_precon));

    InvButterfly<BitShift, false>(&v_X, &v_Y, v_W, v_W_precon, v_neg_modulus,
                                  v_twice_mod);

    WriteInvInterleavedT4(v_X, v_Y, v_X_pt);
    X += 16;

    W += 2;
    W_precon += 2;
  }
}

template <int BitShift>
void InvT8(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t t, uint64_t m, const uint64_t* W,
           const uint64_t* W_precon) {
  size_t j1 = 0;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < m; i++) {
    uint64_t* X = operand + j1;
    uint64_t* Y = X + t;

    __m512i v_W = _mm512_set1_epi64(static_cast<int64_t>(*W++));
    __m512i v_W_precon = _mm512_set1_epi64(static_cast<int64_t>(*W_precon++));

    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
    __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

    // assume 8 | t
    for (size_t j = t / 8; j > 0; --j) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);
      __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

      InvButterfly<BitShift, false>(&v_X, &v_Y, v_W, v_W_precon, v_neg_modulus,
                                    v_twice_mod);

      _mm512_storeu_si512(v_X_pt++, v_X);
      _mm512_storeu_si512(v_Y_pt++, v_Y);
    }
    j1 += (t << 1);
  }
}

template <int BitShift>
void InverseTransformFromBitReverseAVX512(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half) {
  HEXL_CHECK(NTT::CheckArguments(n, modulus), "");
  HEXL_CHECK(n >= 16,
             "InverseTransformFromBitReverseAVX512 doesn't support small "
             "transforms. Need n >= 16, got n = "
                 << n);
  HEXL_CHECK(modulus < NTT::s_max_inv_modulus(BitShift),
             "modulus " << modulus << " too large for BitShift " << BitShift
                        << " => maximum value "
                        << NTT::s_max_inv_modulus(BitShift));
  HEXL_CHECK_BOUNDS(precon_inv_root_of_unity_powers, n, MaximumValue(BitShift),
                    "precon_inv_root_of_unity_powers too large");
  HEXL_CHECK_BOUNDS(operand, n, MaximumValue(BitShift), "operand too large");
  // Skip input bound checking for recursive steps
  HEXL_CHECK_BOUNDS(operand, (recursion_depth == 0) ? n : 0,
                    input_mod_factor * modulus,
                    "operand larger than input_mod_factor * modulus ("
                        << input_mod_factor << " * " << modulus << ")");
  HEXL_CHECK(input_mod_factor == 1 || input_mod_factor == 2,
             "input_mod_factor must be 1 or 2; got " << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2; got " << output_mod_factor);

  uint64_t twice_mod = modulus << 1;
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_neg_modulus = _mm512_set1_epi64(-static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(twice_mod));

  size_t t = 1;
  size_t m = (n >> 1);
  size_t W_idx = 1 + m * recursion_half;

  static const size_t base_ntt_size = 1024;

  if (n <= base_ntt_size) {  // Perform breadth-first InvNTT
    if (operand != result) {
      std::memcpy(result, operand, n * sizeof(uint64_t));
    }

    // Extract t=1, t=2, t=4 loops separately
    {
      // t = 1
      const uint64_t* W = &inv_root_of_unity_powers[W_idx];
      const uint64_t* W_precon = &precon_inv_root_of_unity_powers[W_idx];
      if ((input_mod_factor == 1) && (recursion_depth == 0)) {
        InvT1<BitShift, true>(result, v_neg_modulus, v_twice_mod, m, W,
                              W_precon);
      } else {
        InvT1<BitShift, false>(result, v_neg_modulus, v_twice_mod, m, W,
                               W_precon);
      }

      t <<= 1;
      m >>= 1;
      uint64_t W_idx_delta =
          m * ((1ULL << (recursion_depth + 1)) - recursion_half);
      W_idx += W_idx_delta;

      // t = 2
      W = &inv_root_of_unity_powers[W_idx];
      W_precon = &precon_inv_root_of_unity_powers[W_idx];
      InvT2<BitShift>(result, v_neg_modulus, v_twice_mod, m, W, W_precon);

      t <<= 1;
      m >>= 1;
      W_idx_delta >>= 1;
      W_idx += W_idx_delta;

      // t = 4
      W = &inv_root_of_unity_powers[W_idx];
      W_precon = &precon_inv_root_of_unity_powers[W_idx];
      InvT4<BitShift>(result, v_neg_modulus, v_twice_mod, m, W, W_precon);
      t <<= 1;
      m >>= 1;
      W_idx_delta >>= 1;
      W_idx += W_idx_delta;

      // t >= 8
      for (; m > 1;) {
        W = &inv_root_of_unity_powers[W_idx];
        W_precon = &precon_inv_root_of_unity_powers[W_idx];
        InvT8<BitShift>(result, v_neg_modulus, v_twice_mod, t, m, W, W_precon);
        t <<= 1;
        m >>= 1;
        W_idx_delta >>= 1;
        W_idx += W_idx_delta;
      }
    }
  } else {
    InverseTransformFromBitReverseAVX512<BitShift>(
        result, operand, n / 2, modulus, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor,
        recursion_depth + 1, 2 * recursion_half);
    InverseTransformFromBitReverseAVX512<BitShift>(
        &result[n / 2], &operand[n / 2], n / 2, modulus,
        inv_root_of_unity_powers, precon_inv_root_of_unity_powers,
        input_mod_factor, output_mod_factor, recursion_depth + 1,
        2 * recursion_half + 1);

    uint64_t W_idx_delta =
        m * ((1ULL << (recursion_depth + 1)) - recursion_half);
    for (; m > 2; m >>= 1) {
      t <<= 1;
      W_idx_delta >>= 1;
      W_idx += W_idx_delta;
    }
    if (m == 2) {
      const uint64_t* W = &inv_root_of_unity_powers[W_idx];
      const uint64_t* W_precon = &precon_inv_root_of_unity_powers[W_idx];
      InvT8<BitShift>(result, v_neg_modulus, v_twice_mod, t, m, W, W_precon);
      t <<= 1;
      m >>= 1;
      W_idx_delta >>= 1;
      W_idx += W_idx_delta;
    }
  }

  // Final loop through data
  if (recursion_depth == 0) {
    HEXL_VLOG(4, "AVX512 intermediate result "
                     << std::vector<uint64_t>(result, result + n));

    const uint64_t W = inv_root_of_unity_powers[W_idx];
    MultiplyFactor mf_inv_n(InverseMod(n, modulus), BitShift, modulus);
    const uint64_t inv_n = mf_inv_n.Operand();
    const uint64_t inv_n_prime = mf_inv_n.BarrettFactor();

    MultiplyFactor mf_inv_n_w(MultiplyMod(inv_n, W, modulus), BitShift,
                              modulus);
    const uint64_t inv_n_w = mf_inv_n_w.Operand();
    const uint64_t inv_n_w_prime = mf_inv_n_w.BarrettFactor();

    HEXL_VLOG(4, "inv_n_w " << inv_n_w);

    uint64_t* X = result;
    uint64_t* Y = X + (n >> 1);

    __m512i v_inv_n = _mm512_set1_epi64(static_cast<int64_t>(inv_n));
    __m512i v_inv_n_prime =
        _mm512_set1_epi64(static_cast<int64_t>(inv_n_prime));
    __m512i v_inv_n_w = _mm512_set1_epi64(static_cast<int64_t>(inv_n_w));
    __m512i v_inv_n_w_prime =
        _mm512_set1_epi64(static_cast<int64_t>(inv_n_w_prime));

    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
    __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

    // Merge final InvNTT loop with modulus reduction baked-in
    HEXL_LOOP_UNROLL_4
    for (size_t j = n / 16; j > 0; --j) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);
      __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

      // Slightly different from regular InvButterfly because different W is
      // used for X and Y
      __m512i Y_minus_2q = _mm512_sub_epi64(v_Y, v_twice_mod);
      __m512i X_plus_Y_mod2q =
          _mm512_hexl_small_add_mod_epi64(v_X, v_Y, v_twice_mod);
      // T = *X + twice_mod - *Y
      __m512i T = _mm512_sub_epi64(v_X, Y_minus_2q);

      if (BitShift == 32) {
        __m512i Q1 = _mm512_hexl_mullo_epi<64>(v_inv_n_prime, X_plus_Y_mod2q);
        Q1 = _mm512_srli_epi64(Q1, 32);
        // X = inv_N * X_plus_Y_mod2q - Q1 * modulus;
        __m512i inv_N_tx = _mm512_hexl_mullo_epi<64>(v_inv_n, X_plus_Y_mod2q);
        v_X = _mm512_hexl_mullo_add_lo_epi<64>(inv_N_tx, Q1, v_neg_modulus);

        __m512i Q2 = _mm512_hexl_mullo_epi<64>(v_inv_n_w_prime, T);
        Q2 = _mm512_srli_epi64(Q2, 32);

        // Y = inv_N_W * T - Q2 * modulus;
        __m512i inv_N_W_T = _mm512_hexl_mullo_epi<64>(v_inv_n_w, T);
        v_Y = _mm512_hexl_mullo_add_lo_epi<64>(inv_N_W_T, Q2, v_neg_modulus);
      } else {
        __m512i Q1 =
            _mm512_hexl_mulhi_epi<BitShift>(v_inv_n_prime, X_plus_Y_mod2q);
        // X = inv_N * X_plus_Y_mod2q - Q1 * modulus;
        __m512i inv_N_tx =
            _mm512_hexl_mullo_epi<BitShift>(v_inv_n, X_plus_Y_mod2q);
        v_X =
            _mm512_hexl_mullo_add_lo_epi<BitShift>(inv_N_tx, Q1, v_neg_modulus);

        __m512i Q2 = _mm512_hexl_mulhi_epi<BitShift>(v_inv_n_w_prime, T);
        // Y = inv_N_W * T - Q2 * modulus;
        __m512i inv_N_W_T = _mm512_hexl_mullo_epi<BitShift>(v_inv_n_w, T);
        v_Y = _mm512_hexl_mullo_add_lo_epi<BitShift>(inv_N_W_T, Q2,
                                                     v_neg_modulus);
      }

      if (output_mod_factor == 1) {
        // Modulus reduction from [0, 2q), to [0, q)
        v_X = _mm512_hexl_small_mod_epu64(v_X, v_modulus);
        v_Y = _mm512_hexl_small_mod_epu64(v_Y, v_modulus);
      }

      _mm512_storeu_si512(v_X_pt++, v_X);
      _mm512_storeu_si512(v_Y_pt++, v_Y);
    }

    HEXL_VLOG(5, "AVX512 returning result "
                     << std::vector<uint64_t>(result, result + n));
  }
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
