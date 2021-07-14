// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ntt/inv-ntt-avx512.hpp"

#include "hexl/ntt/ntt.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512IFMA
template void InverseTransformFromBitReverseAVX512<NTT::s_ifma_shift_bits>(
    uint64_t* operand, uint64_t degree, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth = 0,
    uint64_t recursion_half = 0);
#endif

#ifdef HEXL_HAS_AVX512DQ
template void InverseTransformFromBitReverseAVX512<32>(
    uint64_t* operand, uint64_t degree, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth = 0,
    uint64_t recursion_half = 0);

template void InverseTransformFromBitReverseAVX512<NTT::s_default_shift_bits>(
    uint64_t* operand, uint64_t degree, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth = 0,
    uint64_t recursion_half = 0);
#endif

#ifdef HEXL_HAS_AVX512DQ

/// @brief The Harvey butterfly: assume X, Y in [0, 2q), and return X', Y' in
/// [0, 2q). such that X', Y' = X + Y (mod q), W(X - Y) (mod q).
/// @param[in,out] X Input representing 8 64-bit signed integers in SIMD form
/// @param[in,out] Y Input representing 8 64-bit signed integers in SIMD form
/// @param[in] W_op Root of unity representing 8 64-bit signed integers in SIMD
/// form
/// @param[in] W_precon Preconditioned \p W_op for BitShift-bit Barrett
/// reduction
/// @param[in] neg_modulus Negative modulus, i.e. (-q) represented as 8 64-bit
/// signed integers in SIMD form
/// @param[in] twice_modulus Twice the modulus, i.e. 2*q represented as 8 64-bit
/// signed integers in SIMD form
/// @param InputLessThanMod If true, assumes \p X, \p Y < \p q. Otherwise,
/// assumes \p X, \p Y < 2*\p q
/// @details See Algorithm 3 of https://arxiv.org/pdf/1205.2926.pdf
template <int BitShift, bool InputLessThanMod>
inline void InvButterfly(__m512i* X, __m512i* Y, __m512i W_op, __m512i W_precon,
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
    *Y = _mm512_hexl_mullo_add_lo_epi<64>(Q_p, W_op, T);
  } else {
    __m512i Q = _mm512_hexl_mulhi_epi<BitShift>(W_precon, T);
    __m512i Q_p = _mm512_hexl_mullo_epi<BitShift>(Q, neg_modulus);
    *Y = _mm512_hexl_mullo_add_lo_epi<BitShift>(Q_p, W_op, T);
  }
}

template <int BitShift, bool InputLessThanMod>
void InvT1(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  const __m512i* v_W_op_pt = reinterpret_cast<const __m512i*>(W_op);
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

    __m512i v_W_op = _mm512_loadu_si512(v_W_op_pt++);
    __m512i v_W_precon = _mm512_loadu_si512(v_W_precon_pt++);

    InvButterfly<BitShift, InputLessThanMod>(&v_X, &v_Y, v_W_op, v_W_precon,
                                             v_neg_modulus, v_twice_mod);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_X_pt, v_Y);

    j1 += 16;
  }
}

template <int BitShift>
void InvT2(uint64_t* X, __m512i v_neg_modulus, __m512i v_twice_mod, uint64_t m,
           const uint64_t* W_op, const uint64_t* W_precon) {
  // 4 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = m / 4; i > 0; --i) {
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadInvInterleavedT2(X, &v_X, &v_Y);

    __m512i v_W_op = LoadWOpT2(static_cast<const void*>(W_op));
    __m512i v_W_precon = LoadWOpT2(static_cast<const void*>(W_precon));

    InvButterfly<BitShift, false>(&v_X, &v_Y, v_W_op, v_W_precon, v_neg_modulus,
                                  v_twice_mod);

    _mm512_storeu_si512(v_X_pt++, v_X);
    _mm512_storeu_si512(v_X_pt, v_Y);
    X += 16;

    W_op += 4;
    W_precon += 4;
  }
}

template <int BitShift>
void InvT4(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t m, const uint64_t* W_op, const uint64_t* W_precon) {
  uint64_t* X = operand;

  // 2 | m guaranteed by n >= 16
  HEXL_LOOP_UNROLL_4
  for (size_t i = m / 2; i > 0; --i) {
    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);

    __m512i v_X;
    __m512i v_Y;
    LoadInvInterleavedT4(X, &v_X, &v_Y);

    __m512i v_W_op = LoadWOpT4(static_cast<const void*>(W_op));
    __m512i v_W_precon = LoadWOpT4(static_cast<const void*>(W_precon));

    InvButterfly<BitShift, false>(&v_X, &v_Y, v_W_op, v_W_precon, v_neg_modulus,
                                  v_twice_mod);

    WriteInvInterleavedT4(v_X, v_Y, v_X_pt);
    X += 16;

    W_op += 2;
    W_precon += 2;
  }
}

template <int BitShift>
void InvT8(uint64_t* operand, __m512i v_neg_modulus, __m512i v_twice_mod,
           uint64_t t, uint64_t m, const uint64_t* W_op,
           const uint64_t* W_precon) {
  size_t j1 = 0;

  HEXL_LOOP_UNROLL_4
  for (size_t i = 0; i < m; i++) {
    uint64_t* X = operand + j1;
    uint64_t* Y = X + t;

    __m512i v_W_op = _mm512_set1_epi64(static_cast<int64_t>(*W_op++));
    __m512i v_W_precon = _mm512_set1_epi64(static_cast<int64_t>(*W_precon++));

    __m512i* v_X_pt = reinterpret_cast<__m512i*>(X);
    __m512i* v_Y_pt = reinterpret_cast<__m512i*>(Y);

    // assume 8 | t
    for (size_t j = t / 8; j > 0; --j) {
      __m512i v_X = _mm512_loadu_si512(v_X_pt);
      __m512i v_Y = _mm512_loadu_si512(v_Y_pt);

      InvButterfly<BitShift, false>(&v_X, &v_Y, v_W_op, v_W_precon,
                                    v_neg_modulus, v_twice_mod);

      _mm512_storeu_si512(v_X_pt++, v_X);
      _mm512_storeu_si512(v_Y_pt++, v_Y);
    }
    j1 += (t << 1);
  }
}

/// @brief AVX512 implementation of the inverse NTT
/// @param[in, out] operand Input data. Overwritten with NTT output
/// @param[in] n Size of the transfrom, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] modulus Prime modulus. Must satisfy q == 1 mod 2n
/// @param[in] root_of_unity_powers Powers of inverse 2n'th root of unity in
/// F_q. In bit-reversed order.
/// @param[in] precon_root_of_unity_powers Pre-conditioned powers of inverse
/// 2n'th root of unity in F_q. In bit-reversed order.
/// @param[in] input_mod_factor Upper bound for inputs; inputs must be in [0,
/// input_mod_factor * modulus)
/// @param[in] output_mod_factor Upper bound for result; result must be in [0,
/// output_mod_factor * modulus)
/// @param[in] recursion_depth Depth of recursive call
/// @param[in] recursion_half Helper for indexing roots of unity
/// @details The implementation is recursive. The base case is a breadth-first
/// NTT, where all the butterflies in a given stage are processed before any
/// butteflies in the next stage. The base case is small enough to fit in the
/// smallest cache. Larger NTTs are processed recursively in a depth-first
/// manner, such that an entire subtransform is completed before moving to the
/// next subtransform. This reduces the number of cache misses, improving
/// performance on larger transform sizes.
template <int BitShift>
void InverseTransformFromBitReverseAVX512(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half) {
  HEXL_CHECK(CheckNTTArguments(n, modulus), "");
  HEXL_CHECK(n >= 16,
             "InverseTransformFromBitReverseAVX512 doesn't support small "
             "transforms. Need n >= 16, got n = "
                 << n);
  HEXL_CHECK(modulus < MaximumValue(BitShift) / 2,
             "modulus " << modulus << " too large for BitShift " << BitShift
                        << " => maximum value " << MaximumValue(BitShift) / 2);
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
    // Extract t=1, t=2, t=4 loops separately
    {
      // t = 1
      const uint64_t* W_op = &inv_root_of_unity_powers[W_idx];
      const uint64_t* W_precon = &precon_inv_root_of_unity_powers[W_idx];
      if (input_mod_factor == 1) {
        InvT1<BitShift, true>(operand, v_neg_modulus, v_twice_mod, m, W_op,
                              W_precon);
      } else {
        InvT1<BitShift, false>(operand, v_neg_modulus, v_twice_mod, m, W_op,
                               W_precon);
      }

      t <<= 1;
      m >>= 1;
      uint64_t W_idx_delta =
          m * ((1ULL << (recursion_depth + 1)) - recursion_half);
      W_idx += W_idx_delta;

      // t = 2
      W_op = &inv_root_of_unity_powers[W_idx];
      W_precon = &precon_inv_root_of_unity_powers[W_idx];
      InvT2<BitShift>(operand, v_neg_modulus, v_twice_mod, m, W_op, W_precon);

      t <<= 1;
      m >>= 1;
      W_idx_delta >>= 1;
      W_idx += W_idx_delta;

      // t = 4
      W_op = &inv_root_of_unity_powers[W_idx];
      W_precon = &precon_inv_root_of_unity_powers[W_idx];
      InvT4<BitShift>(operand, v_neg_modulus, v_twice_mod, m, W_op, W_precon);
      t <<= 1;
      m >>= 1;
      W_idx_delta >>= 1;
      W_idx += W_idx_delta;

      // t >= 8
      for (; m > 1;) {
        W_op = &inv_root_of_unity_powers[W_idx];
        W_precon = &precon_inv_root_of_unity_powers[W_idx];
        InvT8<BitShift>(operand, v_neg_modulus, v_twice_mod, t, m, W_op,
                        W_precon);
        t <<= 1;
        m >>= 1;
        W_idx_delta >>= 1;
        W_idx += W_idx_delta;
      }
    }
  } else {
    InverseTransformFromBitReverseAVX512<BitShift>(
        operand, n / 2, modulus, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor,
        recursion_depth + 1, 2 * recursion_half);
    InverseTransformFromBitReverseAVX512<BitShift>(
        &operand[n / 2], n / 2, modulus, inv_root_of_unity_powers,
        precon_inv_root_of_unity_powers, input_mod_factor, output_mod_factor,
        recursion_depth + 1, 2 * recursion_half + 1);

    uint64_t W_idx_delta =
        m * ((1ULL << (recursion_depth + 1)) - recursion_half);
    for (; m > 2; m >>= 1) {
      t <<= 1;
      W_idx_delta >>= 1;
      W_idx += W_idx_delta;
    }
    if (m == 2) {
      const uint64_t* W_op = &inv_root_of_unity_powers[W_idx];
      const uint64_t* W_precon = &precon_inv_root_of_unity_powers[W_idx];
      InvT8<BitShift>(operand, v_neg_modulus, v_twice_mod, t, m, W_op,
                      W_precon);
      t <<= 1;
      m >>= 1;
      W_idx_delta >>= 1;
      W_idx += W_idx_delta;
    }
  }

  // Final loop through data
  if (recursion_depth == 0) {
    HEXL_VLOG(4, "AVX512 intermediate operand "
                     << std::vector<uint64_t>(operand, operand + n));

    const uint64_t W_op = inv_root_of_unity_powers[W_idx];
    MultiplyFactor mf_inv_n(InverseMod(n, modulus), BitShift, modulus);
    const uint64_t inv_n = mf_inv_n.Operand();
    const uint64_t inv_n_prime = mf_inv_n.BarrettFactor();

    MultiplyFactor mf_inv_n_w(MultiplyMod(inv_n, W_op, modulus), BitShift,
                              modulus);
    const uint64_t inv_n_w = mf_inv_n_w.Operand();
    const uint64_t inv_n_w_prime = mf_inv_n_w.BarrettFactor();

    HEXL_VLOG(4, "inv_n_w " << inv_n_w);

    uint64_t* X = operand;
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

    HEXL_VLOG(5, "AVX512 returning operand "
                     << std::vector<uint64_t>(operand, operand + n));
  }
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
