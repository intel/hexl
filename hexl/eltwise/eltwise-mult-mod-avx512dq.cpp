// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>
#include <stdint.h>

#include <limits>

#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"
#include "hexl/util/defines.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

template void EltwiseMultModAVX512Float<1>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);
template void EltwiseMultModAVX512Float<2>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);
template void EltwiseMultModAVX512Float<4>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);

template void EltwiseMultModAVX512DQInt<1>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);
template void EltwiseMultModAVX512DQInt<2>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);
template void EltwiseMultModAVX512DQInt<4>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);

#endif

#ifdef HEXL_HAS_AVX512DQ

template <int ProdRightShift, int InputModFactor, int CoeffCount>
void EltwiseMultModAVX512DQIntLoopUnroll(__m512i* vp_result,
                                         const __m512i* vp_operand1,
                                         const __m512i* vp_operand2,
                                         __m512i v_barr_lo, __m512i v_modulus,
                                         __m512i v_twice_mod) {
  constexpr size_t manual_unroll_factor = 16;
  constexpr size_t avx512_64bit_count = 8;
  constexpr size_t loop_count =
      CoeffCount / (manual_unroll_factor * avx512_64bit_count);

  static_assert(loop_count > 0, "loop_count too small for unrolling");
  static_assert(CoeffCount % (manual_unroll_factor * avx512_64bit_count) == 0,
                "CoeffCount must be a factor of manual_unroll_factor * "
                "avx512_64bit_count");

  HEXL_UNUSED(v_twice_mod);
  HEXL_LOOP_UNROLL_4
  for (size_t i = loop_count; i > 0; --i) {
    __m512i x1 = _mm512_loadu_si512(vp_operand1++);
    __m512i y1 = _mm512_loadu_si512(vp_operand2++);
    __m512i x2 = _mm512_loadu_si512(vp_operand1++);
    __m512i y2 = _mm512_loadu_si512(vp_operand2++);
    __m512i x3 = _mm512_loadu_si512(vp_operand1++);
    __m512i y3 = _mm512_loadu_si512(vp_operand2++);
    __m512i x4 = _mm512_loadu_si512(vp_operand1++);
    __m512i y4 = _mm512_loadu_si512(vp_operand2++);
    __m512i x5 = _mm512_loadu_si512(vp_operand1++);
    __m512i y5 = _mm512_loadu_si512(vp_operand2++);
    __m512i x6 = _mm512_loadu_si512(vp_operand1++);
    __m512i y6 = _mm512_loadu_si512(vp_operand2++);
    __m512i x7 = _mm512_loadu_si512(vp_operand1++);
    __m512i y7 = _mm512_loadu_si512(vp_operand2++);
    __m512i x8 = _mm512_loadu_si512(vp_operand1++);
    __m512i y8 = _mm512_loadu_si512(vp_operand2++);
    __m512i x9 = _mm512_loadu_si512(vp_operand1++);
    __m512i y9 = _mm512_loadu_si512(vp_operand2++);
    __m512i x10 = _mm512_loadu_si512(vp_operand1++);
    __m512i y10 = _mm512_loadu_si512(vp_operand2++);
    __m512i x11 = _mm512_loadu_si512(vp_operand1++);
    __m512i y11 = _mm512_loadu_si512(vp_operand2++);
    __m512i x12 = _mm512_loadu_si512(vp_operand1++);
    __m512i y12 = _mm512_loadu_si512(vp_operand2++);
    __m512i x13 = _mm512_loadu_si512(vp_operand1++);
    __m512i y13 = _mm512_loadu_si512(vp_operand2++);
    __m512i x14 = _mm512_loadu_si512(vp_operand1++);
    __m512i y14 = _mm512_loadu_si512(vp_operand2++);
    __m512i x15 = _mm512_loadu_si512(vp_operand1++);
    __m512i y15 = _mm512_loadu_si512(vp_operand2++);
    __m512i x16 = _mm512_loadu_si512(vp_operand1++);
    __m512i y16 = _mm512_loadu_si512(vp_operand2++);

    x1 = _mm512_hexl_small_mod_epu64<InputModFactor>(x1, v_modulus,
                                                     &v_twice_mod);
    x2 = _mm512_hexl_small_mod_epu64<InputModFactor>(x2, v_modulus,
                                                     &v_twice_mod);
    x3 = _mm512_hexl_small_mod_epu64<InputModFactor>(x3, v_modulus,
                                                     &v_twice_mod);
    x4 = _mm512_hexl_small_mod_epu64<InputModFactor>(x4, v_modulus,
                                                     &v_twice_mod);
    x5 = _mm512_hexl_small_mod_epu64<InputModFactor>(x5, v_modulus,
                                                     &v_twice_mod);
    x6 = _mm512_hexl_small_mod_epu64<InputModFactor>(x6, v_modulus,
                                                     &v_twice_mod);
    x7 = _mm512_hexl_small_mod_epu64<InputModFactor>(x7, v_modulus,
                                                     &v_twice_mod);
    x8 = _mm512_hexl_small_mod_epu64<InputModFactor>(x8, v_modulus,
                                                     &v_twice_mod);
    x9 = _mm512_hexl_small_mod_epu64<InputModFactor>(x9, v_modulus,
                                                     &v_twice_mod);
    x10 = _mm512_hexl_small_mod_epu64<InputModFactor>(x10, v_modulus,
                                                      &v_twice_mod);
    x11 = _mm512_hexl_small_mod_epu64<InputModFactor>(x11, v_modulus,
                                                      &v_twice_mod);
    x12 = _mm512_hexl_small_mod_epu64<InputModFactor>(x12, v_modulus,
                                                      &v_twice_mod);
    x13 = _mm512_hexl_small_mod_epu64<InputModFactor>(x13, v_modulus,
                                                      &v_twice_mod);
    x14 = _mm512_hexl_small_mod_epu64<InputModFactor>(x14, v_modulus,
                                                      &v_twice_mod);
    x15 = _mm512_hexl_small_mod_epu64<InputModFactor>(x15, v_modulus,
                                                      &v_twice_mod);
    x16 = _mm512_hexl_small_mod_epu64<InputModFactor>(x16, v_modulus,
                                                      &v_twice_mod);

    y1 = _mm512_hexl_small_mod_epu64<InputModFactor>(y1, v_modulus,
                                                     &v_twice_mod);
    y2 = _mm512_hexl_small_mod_epu64<InputModFactor>(y2, v_modulus,
                                                     &v_twice_mod);
    y3 = _mm512_hexl_small_mod_epu64<InputModFactor>(y3, v_modulus,
                                                     &v_twice_mod);
    y4 = _mm512_hexl_small_mod_epu64<InputModFactor>(y4, v_modulus,
                                                     &v_twice_mod);
    y5 = _mm512_hexl_small_mod_epu64<InputModFactor>(y5, v_modulus,
                                                     &v_twice_mod);
    y6 = _mm512_hexl_small_mod_epu64<InputModFactor>(y6, v_modulus,
                                                     &v_twice_mod);
    y7 = _mm512_hexl_small_mod_epu64<InputModFactor>(y7, v_modulus,
                                                     &v_twice_mod);
    y8 = _mm512_hexl_small_mod_epu64<InputModFactor>(y8, v_modulus,
                                                     &v_twice_mod);
    y9 = _mm512_hexl_small_mod_epu64<InputModFactor>(y9, v_modulus,
                                                     &v_twice_mod);
    y10 = _mm512_hexl_small_mod_epu64<InputModFactor>(y10, v_modulus,
                                                      &v_twice_mod);
    y11 = _mm512_hexl_small_mod_epu64<InputModFactor>(y11, v_modulus,
                                                      &v_twice_mod);
    y12 = _mm512_hexl_small_mod_epu64<InputModFactor>(y12, v_modulus,
                                                      &v_twice_mod);
    y13 = _mm512_hexl_small_mod_epu64<InputModFactor>(y13, v_modulus,
                                                      &v_twice_mod);
    y14 = _mm512_hexl_small_mod_epu64<InputModFactor>(y14, v_modulus,
                                                      &v_twice_mod);
    y15 = _mm512_hexl_small_mod_epu64<InputModFactor>(y15, v_modulus,
                                                      &v_twice_mod);
    y16 = _mm512_hexl_small_mod_epu64<InputModFactor>(y16, v_modulus,
                                                      &v_twice_mod);

    __m512i zhi1 = _mm512_hexl_mulhi_epi<64>(x1, y1);
    __m512i zhi2 = _mm512_hexl_mulhi_epi<64>(x2, y2);
    __m512i zhi3 = _mm512_hexl_mulhi_epi<64>(x3, y3);
    __m512i zhi4 = _mm512_hexl_mulhi_epi<64>(x4, y4);
    __m512i zhi5 = _mm512_hexl_mulhi_epi<64>(x5, y5);
    __m512i zhi6 = _mm512_hexl_mulhi_epi<64>(x6, y6);
    __m512i zhi7 = _mm512_hexl_mulhi_epi<64>(x7, y7);
    __m512i zhi8 = _mm512_hexl_mulhi_epi<64>(x8, y8);
    __m512i zhi9 = _mm512_hexl_mulhi_epi<64>(x9, y9);
    __m512i zhi10 = _mm512_hexl_mulhi_epi<64>(x10, y10);
    __m512i zhi11 = _mm512_hexl_mulhi_epi<64>(x11, y11);
    __m512i zhi12 = _mm512_hexl_mulhi_epi<64>(x12, y12);
    __m512i zhi13 = _mm512_hexl_mulhi_epi<64>(x13, y13);
    __m512i zhi14 = _mm512_hexl_mulhi_epi<64>(x14, y14);
    __m512i zhi15 = _mm512_hexl_mulhi_epi<64>(x15, y15);
    __m512i zhi16 = _mm512_hexl_mulhi_epi<64>(x16, y16);

    __m512i zlo1 = _mm512_hexl_mullo_epi<64>(x1, y1);
    __m512i zlo2 = _mm512_hexl_mullo_epi<64>(x2, y2);
    __m512i zlo3 = _mm512_hexl_mullo_epi<64>(x3, y3);
    __m512i zlo4 = _mm512_hexl_mullo_epi<64>(x4, y4);
    __m512i zlo5 = _mm512_hexl_mullo_epi<64>(x5, y5);
    __m512i zlo6 = _mm512_hexl_mullo_epi<64>(x6, y6);
    __m512i zlo7 = _mm512_hexl_mullo_epi<64>(x7, y7);
    __m512i zlo8 = _mm512_hexl_mullo_epi<64>(x8, y8);
    __m512i zlo9 = _mm512_hexl_mullo_epi<64>(x9, y9);
    __m512i zlo10 = _mm512_hexl_mullo_epi<64>(x10, y10);
    __m512i zlo11 = _mm512_hexl_mullo_epi<64>(x11, y11);
    __m512i zlo12 = _mm512_hexl_mullo_epi<64>(x12, y12);
    __m512i zlo13 = _mm512_hexl_mullo_epi<64>(x13, y13);
    __m512i zlo14 = _mm512_hexl_mullo_epi<64>(x14, y14);
    __m512i zlo15 = _mm512_hexl_mullo_epi<64>(x15, y15);
    __m512i zlo16 = _mm512_hexl_mullo_epi<64>(x16, y16);

    __m512i c1 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo1, zhi1);
    __m512i c2 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo2, zhi2);
    __m512i c3 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo3, zhi3);
    __m512i c4 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo4, zhi4);
    __m512i c5 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo5, zhi5);
    __m512i c6 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo6, zhi6);
    __m512i c7 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo7, zhi7);
    __m512i c8 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo8, zhi8);
    __m512i c9 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo9, zhi9);
    __m512i c10 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo10, zhi10);
    __m512i c11 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo11, zhi11);
    __m512i c12 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo12, zhi12);
    __m512i c13 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo13, zhi13);
    __m512i c14 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo14, zhi14);
    __m512i c15 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo15, zhi15);
    __m512i c16 = _mm512_hexl_shrdi_epi64<ProdRightShift>(zlo16, zhi16);

    c1 = _mm512_hexl_mulhi_approx_epi<64>(c1, v_barr_lo);
    c2 = _mm512_hexl_mulhi_approx_epi<64>(c2, v_barr_lo);
    c3 = _mm512_hexl_mulhi_approx_epi<64>(c3, v_barr_lo);
    c4 = _mm512_hexl_mulhi_approx_epi<64>(c4, v_barr_lo);
    c5 = _mm512_hexl_mulhi_approx_epi<64>(c5, v_barr_lo);
    c6 = _mm512_hexl_mulhi_approx_epi<64>(c6, v_barr_lo);
    c7 = _mm512_hexl_mulhi_approx_epi<64>(c7, v_barr_lo);
    c8 = _mm512_hexl_mulhi_approx_epi<64>(c8, v_barr_lo);
    c9 = _mm512_hexl_mulhi_approx_epi<64>(c9, v_barr_lo);
    c10 = _mm512_hexl_mulhi_approx_epi<64>(c10, v_barr_lo);
    c11 = _mm512_hexl_mulhi_approx_epi<64>(c11, v_barr_lo);
    c12 = _mm512_hexl_mulhi_approx_epi<64>(c12, v_barr_lo);
    c13 = _mm512_hexl_mulhi_approx_epi<64>(c13, v_barr_lo);
    c14 = _mm512_hexl_mulhi_approx_epi<64>(c14, v_barr_lo);
    c15 = _mm512_hexl_mulhi_approx_epi<64>(c15, v_barr_lo);
    c16 = _mm512_hexl_mulhi_approx_epi<64>(c16, v_barr_lo);

    __m512i vr1 = _mm512_hexl_mullo_epi<64>(c1, v_modulus);
    __m512i vr2 = _mm512_hexl_mullo_epi<64>(c2, v_modulus);
    __m512i vr3 = _mm512_hexl_mullo_epi<64>(c3, v_modulus);
    __m512i vr4 = _mm512_hexl_mullo_epi<64>(c4, v_modulus);
    __m512i vr5 = _mm512_hexl_mullo_epi<64>(c5, v_modulus);
    __m512i vr6 = _mm512_hexl_mullo_epi<64>(c6, v_modulus);
    __m512i vr7 = _mm512_hexl_mullo_epi<64>(c7, v_modulus);
    __m512i vr8 = _mm512_hexl_mullo_epi<64>(c8, v_modulus);
    __m512i vr9 = _mm512_hexl_mullo_epi<64>(c9, v_modulus);
    __m512i vr10 = _mm512_hexl_mullo_epi<64>(c10, v_modulus);
    __m512i vr11 = _mm512_hexl_mullo_epi<64>(c11, v_modulus);
    __m512i vr12 = _mm512_hexl_mullo_epi<64>(c12, v_modulus);
    __m512i vr13 = _mm512_hexl_mullo_epi<64>(c13, v_modulus);
    __m512i vr14 = _mm512_hexl_mullo_epi<64>(c14, v_modulus);
    __m512i vr15 = _mm512_hexl_mullo_epi<64>(c15, v_modulus);
    __m512i vr16 = _mm512_hexl_mullo_epi<64>(c16, v_modulus);

    vr1 = _mm512_sub_epi64(zlo1, vr1);
    vr2 = _mm512_sub_epi64(zlo2, vr2);
    vr3 = _mm512_sub_epi64(zlo3, vr3);
    vr4 = _mm512_sub_epi64(zlo4, vr4);
    vr5 = _mm512_sub_epi64(zlo5, vr5);
    vr6 = _mm512_sub_epi64(zlo6, vr6);
    vr7 = _mm512_sub_epi64(zlo7, vr7);
    vr8 = _mm512_sub_epi64(zlo8, vr8);
    vr9 = _mm512_sub_epi64(zlo9, vr9);
    vr10 = _mm512_sub_epi64(zlo10, vr10);
    vr11 = _mm512_sub_epi64(zlo11, vr11);
    vr12 = _mm512_sub_epi64(zlo12, vr12);
    vr13 = _mm512_sub_epi64(zlo13, vr13);
    vr14 = _mm512_sub_epi64(zlo14, vr14);
    vr15 = _mm512_sub_epi64(zlo15, vr15);
    vr16 = _mm512_sub_epi64(zlo16, vr16);

    vr1 = _mm512_hexl_small_mod_epu64<4>(vr1, v_modulus, &v_twice_mod);
    vr2 = _mm512_hexl_small_mod_epu64<4>(vr2, v_modulus, &v_twice_mod);
    vr3 = _mm512_hexl_small_mod_epu64<4>(vr3, v_modulus, &v_twice_mod);
    vr4 = _mm512_hexl_small_mod_epu64<4>(vr4, v_modulus, &v_twice_mod);
    vr5 = _mm512_hexl_small_mod_epu64<4>(vr5, v_modulus, &v_twice_mod);
    vr6 = _mm512_hexl_small_mod_epu64<4>(vr6, v_modulus, &v_twice_mod);
    vr7 = _mm512_hexl_small_mod_epu64<4>(vr7, v_modulus, &v_twice_mod);
    vr8 = _mm512_hexl_small_mod_epu64<4>(vr8, v_modulus, &v_twice_mod);
    vr9 = _mm512_hexl_small_mod_epu64<4>(vr9, v_modulus, &v_twice_mod);
    vr10 = _mm512_hexl_small_mod_epu64<4>(vr10, v_modulus, &v_twice_mod);
    vr11 = _mm512_hexl_small_mod_epu64<4>(vr11, v_modulus, &v_twice_mod);
    vr12 = _mm512_hexl_small_mod_epu64<4>(vr12, v_modulus, &v_twice_mod);
    vr13 = _mm512_hexl_small_mod_epu64<4>(vr13, v_modulus, &v_twice_mod);
    vr14 = _mm512_hexl_small_mod_epu64<4>(vr14, v_modulus, &v_twice_mod);
    vr15 = _mm512_hexl_small_mod_epu64<4>(vr15, v_modulus, &v_twice_mod);
    vr16 = _mm512_hexl_small_mod_epu64<4>(vr16, v_modulus, &v_twice_mod);

    _mm512_storeu_si512(vp_result++, vr1);
    _mm512_storeu_si512(vp_result++, vr2);
    _mm512_storeu_si512(vp_result++, vr3);
    _mm512_storeu_si512(vp_result++, vr4);
    _mm512_storeu_si512(vp_result++, vr5);
    _mm512_storeu_si512(vp_result++, vr6);
    _mm512_storeu_si512(vp_result++, vr7);
    _mm512_storeu_si512(vp_result++, vr8);
    _mm512_storeu_si512(vp_result++, vr9);
    _mm512_storeu_si512(vp_result++, vr10);
    _mm512_storeu_si512(vp_result++, vr11);
    _mm512_storeu_si512(vp_result++, vr12);
    _mm512_storeu_si512(vp_result++, vr13);
    _mm512_storeu_si512(vp_result++, vr14);
    _mm512_storeu_si512(vp_result++, vr15);
    _mm512_storeu_si512(vp_result++, vr16);
  }
}

/// @brief Algorithm 2 from
/// https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf
template <int BitShift, int InputModFactor>
void EltwiseMultModAVX512DQIntLoopDefault(__m512i* vp_result,
                                          const __m512i* vp_operand1,
                                          const __m512i* vp_operand2,
                                          __m512i v_barr_lo, __m512i v_modulus,
                                          __m512i v_twice_mod, uint64_t n) {
  HEXL_UNUSED(v_twice_mod);

  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_op2 = _mm512_loadu_si512(vp_operand2);

    v_op1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1, v_modulus,
                                                        &v_twice_mod);

    v_op2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2, v_modulus,
                                                        &v_twice_mod);

    // Compute product U
    __m512i v_prod_hi = _mm512_hexl_mulhi_epi<64>(v_op1, v_op2);
    __m512i v_prod_lo = _mm512_hexl_mullo_epi<64>(v_op1, v_op2);

    __m512i c1 = _mm512_hexl_shrdi_epi64<BitShift>(v_prod_lo, v_prod_hi);
    // alpha - beta == 64, so we only need high 64 bits
    // Perform approximate computation of high bits, as described on page
    // 7 of https://arxiv.org/pdf/2003.04510.pdf
    __m512i q_hat = _mm512_hexl_mulhi_approx_epi<64>(c1, v_barr_lo);
    __m512i v_result = _mm512_hexl_mullo_epi<64>(q_hat, v_modulus);
    // Computes result in [0, 4q)
    v_result = _mm512_sub_epi64(v_prod_lo, v_result);

    // Reduce result to [0, q)
    v_result =
        _mm512_hexl_small_mod_epu64<4>(v_result, v_modulus, &v_twice_mod);
    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

/// @brief Algorithm 2 from
/// https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf
template <int InputModFactor>
void EltwiseMultModAVX512DQIntLoopDefault(__m512i* vp_result,
                                          const __m512i* vp_operand1,
                                          const __m512i* vp_operand2,
                                          __m512i v_barr_lo, __m512i v_modulus,
                                          __m512i v_twice_mod, uint64_t n,
                                          uint64_t prod_right_shift) {
  HEXL_UNUSED(v_twice_mod);

  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_op2 = _mm512_loadu_si512(vp_operand2);

    v_op1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1, v_modulus,
                                                        &v_twice_mod);

    v_op2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2, v_modulus,
                                                        &v_twice_mod);

    __m512i v_prod_hi = _mm512_hexl_mulhi_epi<64>(v_op1, v_op2);
    __m512i v_prod_lo = _mm512_hexl_mullo_epi<64>(v_op1, v_op2);

    // c1 = floor(U / 2^{n + beta})
    __m512i c1 = _mm512_hexl_shrdi_epi64(
        v_prod_lo, v_prod_hi, static_cast<unsigned int>(prod_right_shift));

    // alpha - beta == 64, so we only need high 64 bits
    // Perform approximate computation of high bits, as described on page
    // 7 of https://arxiv.org/pdf/2003.04510.pdf
    __m512i q_hat = _mm512_hexl_mulhi_approx_epi<64>(c1, v_barr_lo);
    __m512i v_result = _mm512_hexl_mullo_epi<64>(q_hat, v_modulus);
    // Computes result in [0, 4q)
    v_result = _mm512_sub_epi64(v_prod_lo, v_result);

    // Reduce result to [0, q)
    v_result =
        _mm512_hexl_small_mod_epu64<4>(v_result, v_modulus, &v_twice_mod);
    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

template <int ProdRightShift, int InputModFactor>
void EltwiseMultModAVX512DQIntLoop(__m512i* vp_result,
                                   const __m512i* vp_operand1,
                                   const __m512i* vp_operand2,
                                   __m512i v_barr_lo, __m512i v_modulus,
                                   __m512i v_twice_mod, uint64_t n) {
  switch (n) {
    case 1024:
      EltwiseMultModAVX512DQIntLoopUnroll<ProdRightShift, InputModFactor, 1024>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
          v_twice_mod);
      break;

    case 2048:
      EltwiseMultModAVX512DQIntLoopUnroll<ProdRightShift, InputModFactor, 2048>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
          v_twice_mod);
      break;

    case 4096:
      EltwiseMultModAVX512DQIntLoopUnroll<ProdRightShift, InputModFactor, 4096>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
          v_twice_mod);
      break;

    case 8192:
      EltwiseMultModAVX512DQIntLoopUnroll<ProdRightShift, InputModFactor, 8192>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
          v_twice_mod);
      break;

    case 16384:
      EltwiseMultModAVX512DQIntLoopUnroll<ProdRightShift, InputModFactor,
                                          16384>(vp_result, vp_operand1,
                                                 vp_operand2, v_barr_lo,
                                                 v_modulus, v_twice_mod);
      break;

    case 32768:
      EltwiseMultModAVX512DQIntLoopUnroll<ProdRightShift, InputModFactor,
                                          32768>(vp_result, vp_operand1,
                                                 vp_operand2, v_barr_lo,
                                                 v_modulus, v_twice_mod);
      break;

    default:
      EltwiseMultModAVX512DQIntLoopDefault<ProdRightShift, InputModFactor>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
          v_twice_mod, n);
  }
}

#define ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(ProdRightShift, \
                                                             InputModFactor) \
  case (ProdRightShift): {                                                   \
    EltwiseMultModAVX512DQIntLoop<(ProdRightShift), (InputModFactor)>(       \
        vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,           \
        v_twice_mod, n);                                                     \
    break;                                                                   \
  }

// Algorithm 2 from https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf
template <int InputModFactor>
void EltwiseMultModAVX512DQInt(uint64_t* result, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               uint64_t modulus) {
  HEXL_CHECK(InputModFactor == 1 || InputModFactor == 2 || InputModFactor == 4,
             "Require InputModFactor = 1, 2, or 4")
  HEXL_CHECK(InputModFactor * modulus > (1ULL << 50),
             "Require InputModFactor * modulus > (1ULL << 50)")
  HEXL_CHECK(InputModFactor * modulus < (1ULL << 63),
             "Require InputModFactor * modulus < (1ULL << 63)");
  HEXL_CHECK(modulus < (1ULL << 62), "Require  modulus < (1ULL << 62)");
  HEXL_CHECK_BOUNDS(operand1, n, InputModFactor * modulus,
                    "operand1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(operand2, n, InputModFactor * modulus,
                    "operand2 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseMultModNative<InputModFactor>(result, operand1, operand2, n_mod_8,
                                         modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  constexpr int64_t beta = -2;
  HEXL_CHECK(beta <= -2, "beta must be <= -2 for correctness");
  constexpr int64_t alpha = 62;  // ensures alpha - beta = 64
  uint64_t gamma = Log2(InputModFactor);
  HEXL_UNUSED(gamma);
  HEXL_CHECK(alpha >= gamma + 1, "alpha must be >= gamma + 1 for correctness");

  const uint64_t ceil_log_mod = Log2(modulus) + 1;  // "n" from Algorithm 2
  uint64_t prod_right_shift = ceil_log_mod + beta;

  // Barrett factor "mu"
  // TODO(fboemer): Allow MultiplyFactor to take bit shifts != 64
  HEXL_CHECK(ceil_log_mod + alpha >= 64, "ceil_log_mod + alpha < 64");
  uint64_t barr_lo =
      MultiplyFactor(uint64_t(1) << (ceil_log_mod + alpha - 64), 64, modulus)
          .BarrettFactor();

  __m512i v_barr_lo = _mm512_set1_epi64(static_cast<int64_t>(barr_lo));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(2 * modulus));
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  // Let d be the product operand1 * operand2.
  // To ensure d >> prod_right_shift < (1ULL << 64), we need
  // (input_mod_factor * modulus)^2 >> (prod_right_shift) < (1ULL << 64)
  // This happens when 2*log_2(input_mod_factor) + prod_right_shift - beta < 63
  // If not, we need to reduce the inputs to be less than modulus for
  // correctness. This is less efficient, so we avoid it when possible.
  bool reduce_mod = 2 * Log2(InputModFactor) + prod_right_shift - beta >= 63;

  if (reduce_mod) {
    // Here, we assume beta = -2
    HEXL_CHECK(beta == -2, "beta != -2 may skip some cases");
    // This reduce_mod case happens only when
    // prod_right_shift >= 63 - 2 * log2(input_mod_factor) >= 57.
    // Additionally, modulus < (1ULL << 62) implies
    // prod_right_shift <= 61. So N == 57, 58, 59, 60, 61 are the
    // only cases here.
    switch (prod_right_shift) {
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(57, InputModFactor)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(58, InputModFactor)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(59, InputModFactor)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(60, InputModFactor)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(61, InputModFactor)
      default: {
        HEXL_CHECK(false,
                   "Bad value for prod_right_shift: " << prod_right_shift);
      }
    }
  } else {  // Input mod reduction not required; pass InputModFactor == 1.
    // The template arguments are required for use of _mm512_hexl_shrdi_epi64,
    // which requires a compile-time constant for the shift.
    switch (prod_right_shift) {
      // For prod_right_shift < 50, we should prefer EltwiseMultModAVX512Float
      // or EltwiseMultModAVX512IFMAInt, so we don't generate those special
      // cases here
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(50, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(51, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(52, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(53, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(54, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(55, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(56, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(57, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(58, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(59, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(60, 1)
      ELTWISE_MULT_MOD_AVX512_DQ_INT_PROD_RIGHT_SHIFT_CASE(61, 1)
      default: {
        HEXL_VLOG(2, "calling EltwiseMultModAVX512DQIntLoopDefault");
        EltwiseMultModAVX512DQIntLoopDefault<1>(
            vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
            v_twice_mod, n, prod_right_shift);
      }
    }
  }
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

// From Function 18, page 19 of https://arxiv.org/pdf/1407.3383.pdf
// See also Algorithm 2/3 of
// https://hal.archives-ouvertes.fr/hal-02552673/document
template <int InputModFactor>
inline void EltwiseMultModAVX512FloatLoopDefault(
    __m512i* vp_result, const __m512i* vp_operand1, const __m512i* vp_operand2,
    __m512d v_u, __m512d v_p, __m512i v_modulus, __m512i v_twice_mod,
    uint64_t n) {
  HEXL_UNUSED(v_twice_mod);

  constexpr int round_mode = (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op1 = _mm512_loadu_si512(vp_operand1);
    v_op1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1, v_modulus,
                                                        &v_twice_mod);

    __m512i v_op2 = _mm512_loadu_si512(vp_operand2);
    v_op2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2, v_modulus,
                                                        &v_twice_mod);

    __m512d v_x = _mm512_cvt_roundepu64_pd(v_op1, round_mode);
    __m512d v_y = _mm512_cvt_roundepu64_pd(v_op2, round_mode);

    __m512d v_h = _mm512_mul_pd(v_x, v_y);
    __m512d v_l =
        _mm512_fmsub_pd(v_x, v_y, v_h);     // rounding error; h + l == x * y
    __m512d v_b = _mm512_mul_pd(v_h, v_u);  // ~ (x * y) / p
    __m512d v_c = _mm512_floor_pd(v_b);     // ~ floor(x * y / p)
    __m512d v_d = _mm512_fnmadd_pd(v_c, v_p, v_h);
    __m512d v_g = _mm512_add_pd(v_d, v_l);
    __mmask8 m = _mm512_cmp_pd_mask(v_g, _mm512_setzero_pd(), _CMP_LT_OQ);
    v_g = _mm512_mask_add_pd(v_g, m, v_g, v_p);

    __m512i v_result = _mm512_cvt_roundpd_epu64(v_g, round_mode);

    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

template <int InputModFactor, int CoeffCount>
inline void EltwiseMultModAVX512FloatLoopUnroll(
    __m512i* vp_result, const __m512i* vp_operand1, const __m512i* vp_operand2,
    __m512d v_u, __m512d v_p, __m512i v_modulus, __m512i v_twice_mod) {
  constexpr size_t manual_unroll_factor = 4;
  constexpr size_t avx512_64bit_count = 8;
  constexpr size_t loop_count =
      CoeffCount / (manual_unroll_factor * avx512_64bit_count);

  static_assert(loop_count > 0, "loop_count too small for unrolling");
  static_assert(CoeffCount % (manual_unroll_factor * avx512_64bit_count) == 0,
                "CoeffCount must be a factor of manual_unroll_factor * "
                "avx512_64bit_count");

  constexpr int round_mode = (_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC);

  HEXL_LOOP_UNROLL_4
  for (size_t i = loop_count; i > 0; --i) {
    __m512i op1_1 = _mm512_loadu_si512(vp_operand1++);
    __m512i op1_2 = _mm512_loadu_si512(vp_operand1++);
    __m512i op1_3 = _mm512_loadu_si512(vp_operand1++);
    __m512i op1_4 = _mm512_loadu_si512(vp_operand1++);

    __m512i op2_1 = _mm512_loadu_si512(vp_operand2++);
    __m512i op2_2 = _mm512_loadu_si512(vp_operand2++);
    __m512i op2_3 = _mm512_loadu_si512(vp_operand2++);
    __m512i op2_4 = _mm512_loadu_si512(vp_operand2++);

    op1_1 = _mm512_hexl_small_mod_epu64<InputModFactor>(op1_1, v_modulus,
                                                        &v_twice_mod);
    op1_2 = _mm512_hexl_small_mod_epu64<InputModFactor>(op1_2, v_modulus,
                                                        &v_twice_mod);
    op1_3 = _mm512_hexl_small_mod_epu64<InputModFactor>(op1_3, v_modulus,
                                                        &v_twice_mod);
    op1_4 = _mm512_hexl_small_mod_epu64<InputModFactor>(op1_4, v_modulus,
                                                        &v_twice_mod);

    op2_1 = _mm512_hexl_small_mod_epu64<InputModFactor>(op2_1, v_modulus,
                                                        &v_twice_mod);
    op2_2 = _mm512_hexl_small_mod_epu64<InputModFactor>(op2_2, v_modulus,
                                                        &v_twice_mod);
    op2_3 = _mm512_hexl_small_mod_epu64<InputModFactor>(op2_3, v_modulus,
                                                        &v_twice_mod);
    op2_4 = _mm512_hexl_small_mod_epu64<InputModFactor>(op2_4, v_modulus,
                                                        &v_twice_mod);

    __m512d v_x_1 = _mm512_cvt_roundepu64_pd(op1_1, round_mode);
    __m512d v_x_2 = _mm512_cvt_roundepu64_pd(op1_2, round_mode);
    __m512d v_x_3 = _mm512_cvt_roundepu64_pd(op1_3, round_mode);
    __m512d v_x_4 = _mm512_cvt_roundepu64_pd(op1_4, round_mode);

    __m512d v_y_1 = _mm512_cvt_roundepu64_pd(op2_1, round_mode);
    __m512d v_y_2 = _mm512_cvt_roundepu64_pd(op2_2, round_mode);
    __m512d v_y_3 = _mm512_cvt_roundepu64_pd(op2_3, round_mode);
    __m512d v_y_4 = _mm512_cvt_roundepu64_pd(op2_4, round_mode);

    __m512d v_h_1 = _mm512_mul_pd(v_x_1, v_y_1);
    __m512d v_h_2 = _mm512_mul_pd(v_x_2, v_y_2);
    __m512d v_h_3 = _mm512_mul_pd(v_x_3, v_y_3);
    __m512d v_h_4 = _mm512_mul_pd(v_x_4, v_y_4);

    // ~ (x * y) / p
    __m512d v_b_1 = _mm512_mul_pd(v_h_1, v_u);
    __m512d v_b_2 = _mm512_mul_pd(v_h_2, v_u);
    __m512d v_b_3 = _mm512_mul_pd(v_h_3, v_u);
    __m512d v_b_4 = _mm512_mul_pd(v_h_4, v_u);

    // rounding_ error; h + l == x * y
    __m512d v_l_1 = _mm512_fmsub_pd(v_x_1, v_y_1, v_h_1);
    __m512d v_l_2 = _mm512_fmsub_pd(v_x_2, v_y_2, v_h_2);
    __m512d v_l_3 = _mm512_fmsub_pd(v_x_3, v_y_3, v_h_3);
    __m512d v_l_4 = _mm512_fmsub_pd(v_x_4, v_y_4, v_h_4);

    // ~ floor(_x * y / p)
    __m512d v_c_1 = _mm512_floor_pd(v_b_1);
    __m512d v_c_2 = _mm512_floor_pd(v_b_2);
    __m512d v_c_3 = _mm512_floor_pd(v_b_3);
    __m512d v_c_4 = _mm512_floor_pd(v_b_4);

    __m512d v_d_1 = _mm512_fnmadd_pd(v_c_1, v_p, v_h_1);
    __m512d v_d_2 = _mm512_fnmadd_pd(v_c_2, v_p, v_h_2);
    __m512d v_d_3 = _mm512_fnmadd_pd(v_c_3, v_p, v_h_3);
    __m512d v_d_4 = _mm512_fnmadd_pd(v_c_4, v_p, v_h_4);

    __m512d v_g_1 = _mm512_add_pd(v_d_1, v_l_1);
    __m512d v_g_2 = _mm512_add_pd(v_d_2, v_l_2);
    __m512d v_g_3 = _mm512_add_pd(v_d_3, v_l_3);
    __m512d v_g_4 = _mm512_add_pd(v_d_4, v_l_4);

    __mmask8 m_1 = _mm512_cmp_pd_mask(v_g_1, _mm512_setzero_pd(), _CMP_LT_OQ);
    __mmask8 m_2 = _mm512_cmp_pd_mask(v_g_2, _mm512_setzero_pd(), _CMP_LT_OQ);
    __mmask8 m_3 = _mm512_cmp_pd_mask(v_g_3, _mm512_setzero_pd(), _CMP_LT_OQ);
    __mmask8 m_4 = _mm512_cmp_pd_mask(v_g_4, _mm512_setzero_pd(), _CMP_LT_OQ);

    v_g_1 = _mm512_mask_add_pd(v_g_1, m_1, v_g_1, v_p);
    v_g_2 = _mm512_mask_add_pd(v_g_2, m_2, v_g_2, v_p);
    v_g_3 = _mm512_mask_add_pd(v_g_3, m_3, v_g_3, v_p);
    v_g_4 = _mm512_mask_add_pd(v_g_4, m_4, v_g_4, v_p);

    __m512i v_out_1 = _mm512_cvt_roundpd_epu64(v_g_1, round_mode);
    __m512i v_out_2 = _mm512_cvt_roundpd_epu64(v_g_2, round_mode);
    __m512i v_out_3 = _mm512_cvt_roundpd_epu64(v_g_3, round_mode);
    __m512i v_out_4 = _mm512_cvt_roundpd_epu64(v_g_4, round_mode);

    _mm512_storeu_si512(vp_result++, v_out_1);
    _mm512_storeu_si512(vp_result++, v_out_2);
    _mm512_storeu_si512(vp_result++, v_out_3);
    _mm512_storeu_si512(vp_result++, v_out_4);
  }
}

template <int InputModFactor>
inline void EltwiseMultModAVX512FloatLoop(__m512i* vp_result,
                                          const __m512i* vp_operand1,
                                          const __m512i* vp_operand2,
                                          __m512d v_u, __m512d v_p,
                                          __m512i v_modulus,
                                          __m512i v_twice_mod, uint64_t n) {
  switch (n) {
    case 1024:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 1024>(
          vp_result, vp_operand1, vp_operand2, v_u, v_p, v_modulus,
          v_twice_mod);
      break;

    case 2048:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 2048>(
          vp_result, vp_operand1, vp_operand2, v_u, v_p, v_modulus,
          v_twice_mod);
      break;

    case 4096:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 4096>(
          vp_result, vp_operand1, vp_operand2, v_u, v_p, v_modulus,
          v_twice_mod);
      break;

    case 8192:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 8192>(
          vp_result, vp_operand1, vp_operand2, v_u, v_p, v_modulus,
          v_twice_mod);
      break;

    case 16384:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 16384>(
          vp_result, vp_operand1, vp_operand2, v_u, v_p, v_modulus,
          v_twice_mod);
      break;

    case 32768:
      EltwiseMultModAVX512FloatLoopUnroll<InputModFactor, 32768>(
          vp_result, vp_operand1, vp_operand2, v_u, v_p, v_modulus,
          v_twice_mod);
      break;

    default:
      EltwiseMultModAVX512FloatLoopDefault<InputModFactor>(
          vp_result, vp_operand1, vp_operand2, v_u, v_p, v_modulus, v_twice_mod,
          n);
  }
}

// From Function 18, page 19 of https://arxiv.org/pdf/1407.3383.pdf
// See also Algorithm 2/3 of
// https://hal.archives-ouvertes.fr/hal-02552673/document
template <int InputModFactor>
void EltwiseMultModAVX512Float(uint64_t* result, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               uint64_t modulus) {
  HEXL_CHECK(modulus < MaximumValue(50),
             " modulus " << modulus << " exceeds bound " << MaximumValue(50));
  HEXL_CHECK(modulus > 1, "Require modulus > 1");

  HEXL_CHECK_BOUNDS(operand1, n, InputModFactor * modulus,
                    "operand1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(operand2, n, InputModFactor * modulus,
                    "operand2 exceeds bound " << (InputModFactor * modulus));
  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseMultModNative<InputModFactor>(result, operand1, operand2, n_mod_8,
                                         modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }
  __m512d v_p = _mm512_set1_pd(static_cast<double>(modulus));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(modulus * 2));

  // Add epsilon to ensure u * p >= 1.0
  // See Proposition 13 of https://arxiv.org/pdf/1407.3383.pdf
  double u_bar = (1.0 + std::numeric_limits<double>::epsilon()) /
                 static_cast<double>(modulus);
  __m512d v_u = _mm512_set1_pd(u_bar);

  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  // The implementation without modular reduction of the operands is correct
  // as long as (InputModFactor * modulus)^2 < 2^50 * modulus, i.e.
  // InputModFactor^2 * modulus < 2^50.
  // See function 16 of https://arxiv.org/pdf/1407.3383.pdf.
  bool no_input_reduce_mod =
      (InputModFactor * InputModFactor * modulus) < (1ULL << 50);
  if (no_input_reduce_mod) {
    EltwiseMultModAVX512FloatLoop<1>(vp_result, vp_operand1, vp_operand2, v_u,
                                     v_p, v_modulus, v_twice_mod, n);
  } else {
    EltwiseMultModAVX512FloatLoop<InputModFactor>(vp_result, vp_operand1,
                                                  vp_operand2, v_u, v_p,
                                                  v_modulus, v_twice_mod, n);
  }

  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
