// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>
#include <stdint.h>

#include <limits>

#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"
#include "hexl/util/defines.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512IFMA

template void EltwiseMultModAVX512IFMAInt<1>(uint64_t* result,
                                             const uint64_t* operand1,
                                             const uint64_t* operand2,
                                             uint64_t n, uint64_t modulus);
template void EltwiseMultModAVX512IFMAInt<2>(uint64_t* result,
                                             const uint64_t* operand1,
                                             const uint64_t* operand2,
                                             uint64_t n, uint64_t modulus);
template void EltwiseMultModAVX512IFMAInt<4>(uint64_t* result,
                                             const uint64_t* operand1,
                                             const uint64_t* operand2,
                                             uint64_t n, uint64_t modulus);

template <int BitShift, int InputModFactor, int CoeffCount>
void EltwiseMultModAVX512IFMAIntLoopUnroll(__m512i* vp_result,
                                           const __m512i* vp_operand1,
                                           const __m512i* vp_operand2,
                                           __m512i v_barr_lo, __m512i v_modulus,
                                           __m512i v_neg_mod,
                                           __m512i v_twice_mod) {
  constexpr size_t manual_unroll_factor = 16;
  constexpr size_t avx512_64bit_count = 8;
  constexpr size_t loop_count =
      CoeffCount / (manual_unroll_factor * avx512_64bit_count);

  static_assert(loop_count > 0, "loop_count too small for unrolling");
  static_assert(CoeffCount % (manual_unroll_factor * avx512_64bit_count) == 0,
                "CoeffCount must be a factor of manual_unroll_factor * "
                "avx512_64bit_count");

  constexpr uint64_t N = BitShift;
  constexpr unsigned int Nm1 = static_cast<unsigned int>(N - 1);
  constexpr unsigned int HiShift = static_cast<unsigned int>(53 - N);

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

    __m512i v_prod_hi1 = _mm512_hexl_mulhi_epi<52>(x1, y1);
    __m512i v_prod_hi2 = _mm512_hexl_mulhi_epi<52>(x2, y2);
    __m512i v_prod_hi3 = _mm512_hexl_mulhi_epi<52>(x3, y3);
    __m512i v_prod_hi4 = _mm512_hexl_mulhi_epi<52>(x4, y4);
    __m512i v_prod_hi5 = _mm512_hexl_mulhi_epi<52>(x5, y5);
    __m512i v_prod_hi6 = _mm512_hexl_mulhi_epi<52>(x6, y6);
    __m512i v_prod_hi7 = _mm512_hexl_mulhi_epi<52>(x7, y7);
    __m512i v_prod_hi8 = _mm512_hexl_mulhi_epi<52>(x8, y8);
    __m512i v_prod_hi9 = _mm512_hexl_mulhi_epi<52>(x9, y9);
    __m512i v_prod_hi10 = _mm512_hexl_mulhi_epi<52>(x10, y10);
    __m512i v_prod_hi11 = _mm512_hexl_mulhi_epi<52>(x11, y11);
    __m512i v_prod_hi12 = _mm512_hexl_mulhi_epi<52>(x12, y12);
    __m512i v_prod_hi13 = _mm512_hexl_mulhi_epi<52>(x13, y13);
    __m512i v_prod_hi14 = _mm512_hexl_mulhi_epi<52>(x14, y14);
    __m512i v_prod_hi15 = _mm512_hexl_mulhi_epi<52>(x15, y15);
    __m512i v_prod_hi16 = _mm512_hexl_mulhi_epi<52>(x16, y16);

    __m512i v_prod_lo1 = _mm512_hexl_mullo_epi<52>(x1, y1);
    __m512i v_prod_lo2 = _mm512_hexl_mullo_epi<52>(x2, y2);
    __m512i v_prod_lo3 = _mm512_hexl_mullo_epi<52>(x3, y3);
    __m512i v_prod_lo4 = _mm512_hexl_mullo_epi<52>(x4, y4);
    __m512i v_prod_lo5 = _mm512_hexl_mullo_epi<52>(x5, y5);
    __m512i v_prod_lo6 = _mm512_hexl_mullo_epi<52>(x6, y6);
    __m512i v_prod_lo7 = _mm512_hexl_mullo_epi<52>(x7, y7);
    __m512i v_prod_lo8 = _mm512_hexl_mullo_epi<52>(x8, y8);
    __m512i v_prod_lo9 = _mm512_hexl_mullo_epi<52>(x9, y9);
    __m512i v_prod_lo10 = _mm512_hexl_mullo_epi<52>(x10, y10);
    __m512i v_prod_lo11 = _mm512_hexl_mullo_epi<52>(x11, y11);
    __m512i v_prod_lo12 = _mm512_hexl_mullo_epi<52>(x12, y12);
    __m512i v_prod_lo13 = _mm512_hexl_mullo_epi<52>(x13, y13);
    __m512i v_prod_lo14 = _mm512_hexl_mullo_epi<52>(x14, y14);
    __m512i v_prod_lo15 = _mm512_hexl_mullo_epi<52>(x15, y15);
    __m512i v_prod_lo16 = _mm512_hexl_mullo_epi<52>(x16, y16);

    __m512i c1_lo1 = _mm512_srli_epi64(v_prod_lo1, Nm1);
    __m512i c1_lo2 = _mm512_srli_epi64(v_prod_lo2, Nm1);
    __m512i c1_lo3 = _mm512_srli_epi64(v_prod_lo3, Nm1);
    __m512i c1_lo4 = _mm512_srli_epi64(v_prod_lo4, Nm1);
    __m512i c1_lo5 = _mm512_srli_epi64(v_prod_lo5, Nm1);
    __m512i c1_lo6 = _mm512_srli_epi64(v_prod_lo6, Nm1);
    __m512i c1_lo7 = _mm512_srli_epi64(v_prod_lo7, Nm1);
    __m512i c1_lo8 = _mm512_srli_epi64(v_prod_lo8, Nm1);
    __m512i c1_lo9 = _mm512_srli_epi64(v_prod_lo9, Nm1);
    __m512i c1_lo10 = _mm512_srli_epi64(v_prod_lo10, Nm1);
    __m512i c1_lo11 = _mm512_srli_epi64(v_prod_lo11, Nm1);
    __m512i c1_lo12 = _mm512_srli_epi64(v_prod_lo12, Nm1);
    __m512i c1_lo13 = _mm512_srli_epi64(v_prod_lo13, Nm1);
    __m512i c1_lo14 = _mm512_srli_epi64(v_prod_lo14, Nm1);
    __m512i c1_lo15 = _mm512_srli_epi64(v_prod_lo15, Nm1);
    __m512i c1_lo16 = _mm512_srli_epi64(v_prod_lo16, Nm1);

    __m512i c1_hi1 = _mm512_srli_epi64(v_prod_hi1, HiShift);
    __m512i c1_hi2 = _mm512_srli_epi64(v_prod_hi2, HiShift);
    __m512i c1_hi3 = _mm512_srli_epi64(v_prod_hi3, HiShift);
    __m512i c1_hi4 = _mm512_srli_epi64(v_prod_hi4, HiShift);
    __m512i c1_hi5 = _mm512_srli_epi64(v_prod_hi5, HiShift);
    __m512i c1_hi6 = _mm512_srli_epi64(v_prod_hi6, HiShift);
    __m512i c1_hi7 = _mm512_srli_epi64(v_prod_hi7, HiShift);
    __m512i c1_hi8 = _mm512_srli_epi64(v_prod_hi8, HiShift);
    __m512i c1_hi9 = _mm512_srli_epi64(v_prod_hi9, HiShift);
    __m512i c1_hi10 = _mm512_srli_epi64(v_prod_hi10, HiShift);
    __m512i c1_hi11 = _mm512_srli_epi64(v_prod_hi11, HiShift);
    __m512i c1_hi12 = _mm512_srli_epi64(v_prod_hi12, HiShift);
    __m512i c1_hi13 = _mm512_srli_epi64(v_prod_hi13, HiShift);
    __m512i c1_hi14 = _mm512_srli_epi64(v_prod_hi14, HiShift);
    __m512i c1_hi15 = _mm512_srli_epi64(v_prod_hi15, HiShift);
    __m512i c1_hi16 = _mm512_srli_epi64(v_prod_hi16, HiShift);

    __m512i c1_1 = _mm512_or_epi64(c1_lo1, c1_hi1);
    __m512i c1_2 = _mm512_or_epi64(c1_lo2, c1_hi2);
    __m512i c1_3 = _mm512_or_epi64(c1_lo3, c1_hi3);
    __m512i c1_4 = _mm512_or_epi64(c1_lo4, c1_hi4);
    __m512i c1_5 = _mm512_or_epi64(c1_lo5, c1_hi5);
    __m512i c1_6 = _mm512_or_epi64(c1_lo6, c1_hi6);
    __m512i c1_7 = _mm512_or_epi64(c1_lo7, c1_hi7);
    __m512i c1_8 = _mm512_or_epi64(c1_lo8, c1_hi8);
    __m512i c1_9 = _mm512_or_epi64(c1_lo9, c1_hi9);
    __m512i c1_10 = _mm512_or_epi64(c1_lo10, c1_hi10);
    __m512i c1_11 = _mm512_or_epi64(c1_lo11, c1_hi11);
    __m512i c1_12 = _mm512_or_epi64(c1_lo12, c1_hi12);
    __m512i c1_13 = _mm512_or_epi64(c1_lo13, c1_hi13);
    __m512i c1_14 = _mm512_or_epi64(c1_lo14, c1_hi14);
    __m512i c1_15 = _mm512_or_epi64(c1_lo15, c1_hi15);
    __m512i c1_16 = _mm512_or_epi64(c1_lo16, c1_hi16);

    __m512i c3_1 = _mm512_hexl_mulhi_epi<52>(c1_1, v_barr_lo);
    __m512i c3_2 = _mm512_hexl_mulhi_epi<52>(c1_2, v_barr_lo);
    __m512i c3_3 = _mm512_hexl_mulhi_epi<52>(c1_3, v_barr_lo);
    __m512i c3_4 = _mm512_hexl_mulhi_epi<52>(c1_4, v_barr_lo);
    __m512i c3_5 = _mm512_hexl_mulhi_epi<52>(c1_5, v_barr_lo);
    __m512i c3_6 = _mm512_hexl_mulhi_epi<52>(c1_6, v_barr_lo);
    __m512i c3_7 = _mm512_hexl_mulhi_epi<52>(c1_7, v_barr_lo);
    __m512i c3_8 = _mm512_hexl_mulhi_epi<52>(c1_8, v_barr_lo);
    __m512i c3_9 = _mm512_hexl_mulhi_epi<52>(c1_9, v_barr_lo);
    __m512i c3_10 = _mm512_hexl_mulhi_epi<52>(c1_10, v_barr_lo);
    __m512i c3_11 = _mm512_hexl_mulhi_epi<52>(c1_11, v_barr_lo);
    __m512i c3_12 = _mm512_hexl_mulhi_epi<52>(c1_12, v_barr_lo);
    __m512i c3_13 = _mm512_hexl_mulhi_epi<52>(c1_13, v_barr_lo);
    __m512i c3_14 = _mm512_hexl_mulhi_epi<52>(c1_14, v_barr_lo);
    __m512i c3_15 = _mm512_hexl_mulhi_epi<52>(c1_15, v_barr_lo);
    __m512i c3_16 = _mm512_hexl_mulhi_epi<52>(c1_16, v_barr_lo);

    __m512i c4_1 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo1, c3_1, v_neg_mod);
    __m512i c4_2 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo2, c3_2, v_neg_mod);
    __m512i c4_3 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo3, c3_3, v_neg_mod);
    __m512i c4_4 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo4, c3_4, v_neg_mod);
    __m512i c4_5 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo5, c3_5, v_neg_mod);
    __m512i c4_6 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo6, c3_6, v_neg_mod);
    __m512i c4_7 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo7, c3_7, v_neg_mod);
    __m512i c4_8 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo8, c3_8, v_neg_mod);
    __m512i c4_9 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo9, c3_9, v_neg_mod);
    __m512i c4_10 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo10, c3_10, v_neg_mod);
    __m512i c4_11 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo11, c3_11, v_neg_mod);
    __m512i c4_12 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo12, c3_12, v_neg_mod);
    __m512i c4_13 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo13, c3_13, v_neg_mod);
    __m512i c4_14 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo14, c3_14, v_neg_mod);
    __m512i c4_15 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo15, c3_15, v_neg_mod);
    __m512i c4_16 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo16, c3_16, v_neg_mod);

    __m512i vr1 = _mm512_hexl_small_mod_epu64<2>(c4_1, v_modulus, &v_twice_mod);
    __m512i vr2 = _mm512_hexl_small_mod_epu64<2>(c4_2, v_modulus, &v_twice_mod);
    __m512i vr3 = _mm512_hexl_small_mod_epu64<2>(c4_3, v_modulus, &v_twice_mod);
    __m512i vr4 = _mm512_hexl_small_mod_epu64<2>(c4_4, v_modulus, &v_twice_mod);
    __m512i vr5 = _mm512_hexl_small_mod_epu64<2>(c4_5, v_modulus, &v_twice_mod);
    __m512i vr6 = _mm512_hexl_small_mod_epu64<2>(c4_6, v_modulus, &v_twice_mod);
    __m512i vr7 = _mm512_hexl_small_mod_epu64<2>(c4_7, v_modulus, &v_twice_mod);
    __m512i vr8 = _mm512_hexl_small_mod_epu64<2>(c4_8, v_modulus, &v_twice_mod);
    __m512i vr9 = _mm512_hexl_small_mod_epu64<2>(c4_9, v_modulus, &v_twice_mod);
    __m512i vr10 =
        _mm512_hexl_small_mod_epu64<2>(c4_10, v_modulus, &v_twice_mod);
    __m512i vr11 =
        _mm512_hexl_small_mod_epu64<2>(c4_11, v_modulus, &v_twice_mod);
    __m512i vr12 =
        _mm512_hexl_small_mod_epu64<2>(c4_12, v_modulus, &v_twice_mod);
    __m512i vr13 =
        _mm512_hexl_small_mod_epu64<2>(c4_13, v_modulus, &v_twice_mod);
    __m512i vr14 =
        _mm512_hexl_small_mod_epu64<2>(c4_14, v_modulus, &v_twice_mod);
    __m512i vr15 =
        _mm512_hexl_small_mod_epu64<2>(c4_15, v_modulus, &v_twice_mod);
    __m512i vr16 =
        _mm512_hexl_small_mod_epu64<2>(c4_16, v_modulus, &v_twice_mod);

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

// Algorithm 1 from
// https://hal.archives-ouvertes.fr/hal-01215845/document
template <int BitShift, int InputModFactor>
void EltwiseMultModAVX512IFMAIntLoopDefault(
    __m512i* vp_result, const __m512i* vp_operand1, const __m512i* vp_operand2,
    __m512i v_barr_lo, __m512i v_modulus, __m512i v_neg_mod,
    __m512i v_twice_mod, uint64_t n) {
  uint64_t N = BitShift;

  HEXL_UNUSED(v_twice_mod);
  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    v_operand1 = _mm512_hexl_small_mod_epu64<InputModFactor>(
        v_operand1, v_modulus, &v_twice_mod);

    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);
    v_operand2 = _mm512_hexl_small_mod_epu64<InputModFactor>(
        v_operand2, v_modulus, &v_twice_mod);

    // Compute product
    __m512i v_prod_hi = _mm512_hexl_mulhi_epi<52>(v_operand1, v_operand2);
    __m512i v_prod_lo = _mm512_hexl_mullo_epi<52>(v_operand1, v_operand2);

    __m512i c1_lo =
        _mm512_srli_epi64(v_prod_lo, static_cast<unsigned int>(N - 1ULL));
    __m512i c1_hi = _mm512_slli_epi64(
        v_prod_hi, static_cast<unsigned int>(52ULL - (N - 1ULL)));
    __m512i c1 = _mm512_or_epi64(c1_lo, c1_hi);

    // L - N + 1 == 52, so we only need high 52 bits
    __m512i c3 = _mm512_hexl_mulhi_epi<52>(c1, v_barr_lo);

    // C4 = prod_lo - (p * c3)_lo
    __m512i v_result =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo, c3, v_neg_mod);

    // Reduce result to [0, q)
    v_result =
        _mm512_hexl_small_mod_epu64<2>(v_result, v_modulus, &v_twice_mod);
    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

// Algorithm 1 from
// https://hal.archives-ouvertes.fr/hal-01215845/document
template <int InputModFactor>
void EltwiseMultModAVX512IFMAIntLoopDefault(
    __m512i* vp_result, const __m512i* vp_operand1, const __m512i* vp_operand2,
    __m512i v_barr_lo, __m512i v_modulus, __m512i v_neg_mod,
    __m512i v_twice_mod, uint64_t n, uint64_t bit_shift) {
  uint64_t N = bit_shift;

  HEXL_UNUSED(v_twice_mod);
  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    v_operand1 = _mm512_hexl_small_mod_epu64<InputModFactor>(
        v_operand1, v_modulus, &v_twice_mod);

    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);
    v_operand2 = _mm512_hexl_small_mod_epu64<InputModFactor>(
        v_operand2, v_modulus, &v_twice_mod);

    // Compute product
    __m512i v_prod_hi = _mm512_hexl_mulhi_epi<52>(v_operand1, v_operand2);
    __m512i v_prod_lo = _mm512_hexl_mullo_epi<52>(v_operand1, v_operand2);

    __m512i c1_lo =
        _mm512_srli_epi64(v_prod_lo, static_cast<unsigned int>(N - 1ULL));
    __m512i c1_hi = _mm512_slli_epi64(
        v_prod_hi, static_cast<unsigned int>(52ULL - (N - 1ULL)));
    __m512i c1 = _mm512_or_epi64(c1_lo, c1_hi);

    // L - N + 1 == 52, so we only need high 52 bits
    __m512i c3 = _mm512_hexl_mulhi_epi<52>(c1, v_barr_lo);

    // C4 = prod_lo - (p * c3)_lo
    __m512i v_result =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo, c3, v_neg_mod);

    // Reduce result to [0, q)
    v_result =
        _mm512_hexl_small_mod_epu64<2>(v_result, v_modulus, &v_twice_mod);
    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

template <int BitShift, int InputModFactor>
void EltwiseMultModAVX512IFMAIntLoop(__m512i* vp_result,
                                     const __m512i* vp_operand1,
                                     const __m512i* vp_operand2,
                                     __m512i v_barr_lo, __m512i v_modulus,
                                     __m512i v_neg_mod, __m512i v_twice_mod,
                                     uint64_t n) {
  LOG(INFO) << "EltwiseMultModAVX512IFMAIntLoop BitShift " << BitShift;
  switch (n) {
      // case 1024:
      //   EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor,
      //   1024>(
      //       vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
      //       v_neg_mod, v_twice_mod);
      //   break;

      // case 2048:
      //   EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor,
      //   2048>(
      //       vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
      //       v_neg_mod, v_twice_mod);
      //   break;

      // case 4096:
      //   EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor,
      //   4096>(
      //       vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
      //       v_neg_mod, v_twice_mod);
      //   break;

      // case 8192:
      //   EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor,
      //   8192>(
      //       vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
      //       v_neg_mod, v_twice_mod);
      //   break;

      // case 16384:
      //   EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor,
      //   16384>(
      //       vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
      //       v_neg_mod, v_twice_mod);
      //   break;

      // case 32768:
      //   EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor,
      //   32768>(
      //       vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
      //       v_neg_mod, v_twice_mod);
      //   break;

    default:
      EltwiseMultModAVX512IFMAIntLoopDefault<BitShift, InputModFactor>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod, n);
  }
}

#define GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(BitShift)     \
  case (BitShift): {                                                          \
    EltwiseMultModAVX512IFMAIntLoop<(BitShift), InputModFactor>(              \
        vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod, \
        v_twice_mod, n);                                                      \
    break;                                                                    \
  }

// Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document
template <int InputModFactor>
void EltwiseMultModAVX512IFMAInt(uint64_t* result, const uint64_t* operand1,
                                 const uint64_t* operand2, uint64_t n,
                                 uint64_t modulus) {
  HEXL_CHECK(InputModFactor == 1 || InputModFactor == 2 || InputModFactor == 4,
             "Require InputModFactor = 1, 2, or 4")
  HEXL_CHECK(modulus < (1ULL << 50), "Require  modulus < (1ULL << 50)");
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

  LOG(INFO) << "mod " << modulus;

  const uint64_t logmod = MSB(modulus);

  // modulus < 2**N
  const uint64_t N = logmod + 1;
  uint64_t L = 51 + N;  // Ensures L-N+1 == 52
  uint64_t barr_lo =
      MultiplyFactor(uint64_t(1) << (L - 52), 52, modulus).BarrettFactor();

  __m512i v_barr_lo = _mm512_set1_epi64(static_cast<int64_t>(barr_lo));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(2 * modulus));
  __m512i v_neg_mod = _mm512_set1_epi64(-static_cast<int64_t>(modulus));
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  // The template arguments are required for use of _mm512_hexl_shrdi_epi64,
  // which requires a compile-time constant for the shift.
  switch (N) {
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(40)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(41)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(42)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(43)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(44)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(45)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(46)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(47)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(48)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(49)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(50)
    GENERATE_ELTWISE_MULT_MOD_AVX512_IFMA_INT_BITSHIFT_CASE(51)
    default: {
      EltwiseMultModAVX512IFMAIntLoopDefault<InputModFactor>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod, n, N);
    }
  }
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

#endif

}  // namespace hexl
}  // namespace intel
