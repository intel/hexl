// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-dot-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include "eltwise/eltwise-dot-mod-internal.hpp"
#include "hexl/eltwise/eltwise-dot-mod.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "util/avx512-util.hpp"

#ifdef HEXL_HAS_AVX512DQ

namespace intel {
namespace hexl {

// void EltwiseDotModAVX512(uint64_t* result, const uint64_t* operand1,
//                          const uint64_t* operand2, const uint64_t* operand3,
//                          const uint64_t* operand4, uint64_t n,
//                          uint64_t modulus) {
//   HEXL_CHECK(result != nullptr, "Require result != nullptr");
//   HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
//   HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
//   HEXL_CHECK(operand3 != nullptr, "Require operand3 != nullptr");
//   HEXL_CHECK(operand4 != nullptr, "Require operand4 != nullptr");
//   HEXL_CHECK(n != 0, "Require n != 0");
//   HEXL_CHECK(modulus > 1, "Require modulus > 1");
//   HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
//   HEXL_CHECK_BOUNDS(operand1, n, modulus,
//                     "pre-dot value in operand1 exceeds bound " << modulus);
//   HEXL_CHECK_BOUNDS(operand2, n, modulus,
//                     "pre-dot value in operand1 exceeds bound " << modulus);
//   HEXL_CHECK_BOUNDS(operand3, n, modulus,
//                     "pre-dot value in operand1 exceeds bound " << modulus);
//   HEXL_CHECK_BOUNDS(operand4, n, modulus,
//                     "pre-dot value in operand1 exceeds bound " << modulus)

//   uint64_t n_mod_8 = n % 8;
//   if (n_mod_8 != 0) {
//     EltwiseDotModNative(result, operand1, operand2, operand3, operand4,
//     n_mod_8,
//                         modulus);
//     operand1 += n_mod_8;
//     operand2 += n_mod_8;
//     result += n_mod_8;
//     n -= n_mod_8;
//   }

//   __m512d p = _mm512_set1_pd(static_cast<double>(modulus));
//   __m512i* vp_result = reinterpret_cast<__m512i*>(result);
//   const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
//   const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
//   const __m512i* vp_operand3 = reinterpret_cast<const __m512i*>(operand3);
//   const __m512i* vp_operand4 = reinterpret_cast<const __m512i*>(operand4);

//   // Add epsilon to ensure u * p >= 1.0
//   // See Proposition 13 of https://arxiv.org/pdf/1407.3383.pdf
//   double ubar = (1.0 + std::numeric_limits<double>::epsilon()) /
//                 static_cast<double>(modulus);
//   __m512d u = _mm512_set1_pd(ubar);

//   uint64_t mu = MultiplyFactor(1, 64, modulus).BarrettFactor();
//   __m512i v_mu = _mm512_set1_epi64(static_cast<int64_t>(mu));
//   constexpr int round_mode = (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

//   LOG(INFO) << "modulus " << modulus;

//   HEXL_LOOP_UNROLL_4
//   for (size_t i = n / 8; i > 0; --i) {
//     __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
//     __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);
//     __m512i v_operand3 = _mm512_loadu_si512(vp_operand3);
//     __m512i v_operand4 = _mm512_loadu_si512(vp_operand4);

//     LOG(INFO) << "loaded op1 " << ExtractValues(v_operand1);
//     LOG(INFO) << "loaded op2 " << ExtractValues(v_operand2);
//     LOG(INFO) << "loaded op3 " << ExtractValues(v_operand3);
//     LOG(INFO) << "loaded op4 " << ExtractValues(v_operand4);

//     __m512d x1 = _mm512_cvt_roundepu64_pd(v_operand1, round_mode);
//     __m512d y1 = _mm512_cvt_roundepu64_pd(v_operand2, round_mode);
//     __m512d x2 = _mm512_cvt_roundepu64_pd(v_operand3, round_mode);
//     __m512d y2 = _mm512_cvt_roundepu64_pd(v_operand4, round_mode);

//     // Dot product with no mod
//     __m512d h1 = _mm512_mul_pd(x1, y1);
//     __m512d h2 = _mm512_mul_pd(x2, y2);
//     __m512d h = _mm512_add_pd(h1, h2);

//     LOG(INFO) << "h " << ExtractDoubleValues(h);
//     __m512i hi = _mm512_cvt_roundpd_epu64(h, round_mode);

//     LOG(INFO) << "hi " << ExtractIntValues(hi);

//     // modular reduction
//     // rounding error; h1 + l1 == x1 * y1
//     __m512d l1 = _mm512_fmsub_pd(x1, y1, h1);

//     // rounding error; h2 + l2 == x2 * y2
//     __m512d l2 = _mm512_fmsub_pd(x2, y2, h2);
//     __m512d l = _mm512_add_pd(l1, l2);

//     LOG(INFO) << "l1 " << ExtractDoubleValues(l1);
//     LOG(INFO) << "l2 " << ExtractDoubleValues(l2);
//     LOG(INFO) << "l " << ExtractDoubleValues(l);

//     __m512i li = _mm512_cvt_roundpd_epu64(l, round_mode);
//     __m512i hi_p_l = _mm512_add_epi64(hi, li);
//     LOG(INFO) << "hi + l " << ExtractIntValues(hi_p_l);

//     __m512d b1 = _mm512_mul_pd(h1, u);  // ~ (x1 * y1) / p
//     __m512d b2 = _mm512_mul_pd(h2, u);  // ~ (x2 * y2) / p
//     __m512d c1 = _mm512_floor_pd(b1);   // ~ floor(x1 * y1 / p)
//     __m512d c2 = _mm512_floor_pd(b2);   // ~ floor(x2 * y2 / p)
//     LOG(INFO) << "c1 " << ExtractDoubleValues(c1);
//     LOG(INFO) << "c2 " << ExtractDoubleValues(c2);

//     __m512d c = _mm512_add_pd(c1, c2);

//     __m512i ci = _mm512_cvt_roundpd_epu64(c, round_mode);
//     LOG(INFO) << "ci " << ExtractIntValues(ci);

//     __m512d d1 = _mm512_fnmadd_pd(c, p, h1);
//     __m512d d2 = _mm512_fnmadd_pd(c, p, h2);

//     LOG(INFO) << "d1 " << ExtractDoubleValues(d1);

//     // if g < 0, g >= p, g -= p
//     __m512d g1 = _mm512_add_pd(d1, l1);
//     __mmask8 m1 = _mm512_cmp_pd_mask(g1, _mm512_setzero_pd(), _CMP_LT_OQ);
//     g1 = _mm512_mask_add_pd(g1, m1, g1, p);

//     // if g < 0, g >= p, g -= p
//     __m512d g2 = _mm512_add_pd(d2, l2);
//     __mmask8 m2 = _mm512_cmp_pd_mask(g2, _mm512_setzero_pd(), _CMP_LT_OQ);
//     g2 = _mm512_mask_add_pd(g2, m2, g2, p);

//     __m512d g = _mm512_add_pd(g1, g2);

//     __m512i v_result = _mm512_cvt_roundpd_epu64(g, round_mode);

//     _mm512_storeu_si512(vp_result, v_result);

//     LOG(INFO) << "v_result " << ExtractValues(v_result);

//     ++vp_operand1;
//     ++vp_operand2;
//     ++vp_operand3;
//     ++vp_operand4;
//     ++vp_result;
//   }
//   HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
// }

void EltwiseDotModAVX512(uint64_t* result, const uint64_t** operand1,
                         const uint64_t** operand2, uint64_t num_vectors,
                         uint64_t n, uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(n % 8 == 0, "Require n %8 == 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1[0], n, modulus,
                    "pre-dot value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2[0], n, modulus,
                    "pre-dot value in operand1 exceeds bound " << modulus);

  // Compute intermediate sums
  // LOG(INFO) << "EltwiseDotModAVX512 num_vectors " << num_vectors;
  // LOG(INFO) << "EltwiseDotModAVX512 n " << n << " p " << modulus;

  AlignedVector64<uint64_t> sum_hi(n, 0);
  AlignedVector64<uint64_t> sum_lo(n, 0);

  __m512i* v_sum_hi = reinterpret_cast<__m512i*>(&sum_hi[0]);
  __m512i* v_sum_lo = reinterpret_cast<__m512i*>(&sum_lo[0]);
  // std::vector<__m512i> sum_hi(n / 8, _mm512_set1_epi64(0));
  // std::vector<__m512i> sum_lo(n / 8, _mm512_set1_epi64(0));
  HEXL_LOOP_UNROLL_4
  for (size_t k = 0; k < num_vectors; ++k) {
    const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1[k]);
    const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2[k]);

    HEXL_LOOP_UNROLL_4
    for (size_t i = 0; i < n / 8; ++i) {
      __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
      __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

      // LOG(INFO) << "Loaded " << ExtractValues(v_operand1);
      // LOG(INFO) << "Loaded " << ExtractValues(v_operand2);

      __m512i vprod_hi = _mm512_hexl_mulhi_epi<64>(v_operand1, v_operand2);
      __m512i vprod_lo = _mm512_hexl_mullo_epi<64>(v_operand1, v_operand2);

      _mm512_hexl_add_epi128(vprod_hi, vprod_lo, v_sum_hi[i], v_sum_lo[i],
                             &v_sum_hi[i], &v_sum_lo[i]);

      // LOG(INFO) << "added hi " << ExtractValues(v_sum_hi[i]);
      // LOG(INFO) << "added lo " << ExtractValues(v_sum_lo[i]);
    }
  }

  // Compute Barrett reduction on the inputs

  const uint64_t logmod = MSB(modulus);

  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  // modulus < 2**N
  const uint64_t N = logmod + 1;
  uint64_t L = 63 + N;  // Ensures L-N+1 == 64
  uint64_t op_hi = uint64_t(1) << (L - 64);
  uint64_t op_lo = uint64_t(0);
  uint64_t barr_lo = DivideUInt128UInt64Lo(op_hi, op_lo, modulus);

  // Let d be the product operand1 * operand2.
  // To ensure d >> (N - 1) < (1ULL << 64), we need
  // (input_mod_factor * modulus)^2 >> (N-1) < (1ULL << 64)
  // This happens when 2 * log_2(input_mod_factor) + N < 63
  // If not, we need to reduce the inputs to be less than modulus for
  // correctness. This is less efficient, so we avoid it when possible.
  // bool reduce_mod = 2 * log2_input_mod_factor + N >= 63;

  __m512i vbarr_lo = _mm512_set1_epi64(static_cast<int64_t>(barr_lo));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(2 * modulus));

  // LOG(INFO) << "reducing sum\n";

  // LOG(INFO) << "L " << L;

  HEXL_LOOP_UNROLL_8
  for (size_t i = 0; i < n / 8; ++i) {
    // LOG(INFO) << "loaded v_sum_lo " << ExtractValues(v_sum_lo[i]);
    // LOG(INFO) << "loaded v_sum_hi " << ExtractValues(v_sum_hi[i]);
    __m512i c1 = _mm512_hexl_shrdi_epi64(v_sum_lo[i], v_sum_hi[i],
                                         static_cast<unsigned int>(N - 1));
    // LOG(INFO) << "N " << N;
    // LOG(INFO) << "c1 " << ExtractValues(c1);

    // L - N + 1 == 64, so we only need high 64 bits
    __m512i c3 = _mm512_hexl_mulhi_epi<64>(c1, vbarr_lo);

    // C4 = prod_lo - (p * c3)_lo
    __m512i vresult = _mm512_hexl_mullo_epi<64>(c3, v_modulus);
    vresult = _mm512_sub_epi64(v_sum_lo[i], vresult);

    // Conditional subtraction
    vresult = _mm512_hexl_small_mod_epu64(vresult, v_modulus);

    // LOG(INFO) << "vresult " << ExtractValues(vresult);
    _mm512_storeu_si512(vp_result, vresult);
    vp_result++;
  }
}

}  // namespace hexl
}  // namespace intel

#endif
