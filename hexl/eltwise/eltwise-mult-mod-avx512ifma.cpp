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
    __m512i v_op1_1 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_1 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_2 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_2 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_3 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_3 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_4 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_4 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_5 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_5 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_6 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_6 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_7 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_7 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_8 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_8 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_9 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_9 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_10 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_10 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_11 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_11 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_12 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_12 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_13 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_13 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_14 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_14 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_15 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_15 = _mm512_loadu_si512(vp_operand2++);
    __m512i v_op1_16 = _mm512_loadu_si512(vp_operand1++);
    __m512i v_op2_16 = _mm512_loadu_si512(vp_operand2++);

    v_op1_1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_1, v_modulus,
                                                          &v_twice_mod);
    v_op1_2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_2, v_modulus,
                                                          &v_twice_mod);
    v_op1_3 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_3, v_modulus,
                                                          &v_twice_mod);
    v_op1_4 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_4, v_modulus,
                                                          &v_twice_mod);
    v_op1_5 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_5, v_modulus,
                                                          &v_twice_mod);
    v_op1_6 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_6, v_modulus,
                                                          &v_twice_mod);
    v_op1_7 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_7, v_modulus,
                                                          &v_twice_mod);
    v_op1_8 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_8, v_modulus,
                                                          &v_twice_mod);
    v_op1_9 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_9, v_modulus,
                                                          &v_twice_mod);
    v_op1_10 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_10, v_modulus,
                                                           &v_twice_mod);
    v_op1_11 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_11, v_modulus,
                                                           &v_twice_mod);
    v_op1_12 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_12, v_modulus,
                                                           &v_twice_mod);
    v_op1_13 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_13, v_modulus,
                                                           &v_twice_mod);
    v_op1_14 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_14, v_modulus,
                                                           &v_twice_mod);
    v_op1_15 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_15, v_modulus,
                                                           &v_twice_mod);
    v_op1_16 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1_16, v_modulus,
                                                           &v_twice_mod);

    v_op2_1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_1, v_modulus,
                                                          &v_twice_mod);
    v_op2_2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_2, v_modulus,
                                                          &v_twice_mod);
    v_op2_3 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_3, v_modulus,
                                                          &v_twice_mod);
    v_op2_4 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_4, v_modulus,
                                                          &v_twice_mod);
    v_op2_5 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_5, v_modulus,
                                                          &v_twice_mod);
    v_op2_6 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_6, v_modulus,
                                                          &v_twice_mod);
    v_op2_7 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_7, v_modulus,
                                                          &v_twice_mod);
    v_op2_8 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_8, v_modulus,
                                                          &v_twice_mod);
    v_op2_9 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_9, v_modulus,
                                                          &v_twice_mod);
    v_op2_10 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_10, v_modulus,
                                                           &v_twice_mod);
    v_op2_11 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_11, v_modulus,
                                                           &v_twice_mod);
    v_op2_12 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_12, v_modulus,
                                                           &v_twice_mod);
    v_op2_13 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_13, v_modulus,
                                                           &v_twice_mod);
    v_op2_14 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_14, v_modulus,
                                                           &v_twice_mod);
    v_op2_15 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_15, v_modulus,
                                                           &v_twice_mod);
    v_op2_16 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2_16, v_modulus,
                                                           &v_twice_mod);

    __m512i v_prod_hi_1 = _mm512_hexl_mulhi_epi<52>(v_op1_1, v_op2_1);
    __m512i v_prod_hi_2 = _mm512_hexl_mulhi_epi<52>(v_op1_2, v_op2_2);
    __m512i v_prod_hi_3 = _mm512_hexl_mulhi_epi<52>(v_op1_3, v_op2_3);
    __m512i v_prod_hi_4 = _mm512_hexl_mulhi_epi<52>(v_op1_4, v_op2_4);
    __m512i v_prod_hi_5 = _mm512_hexl_mulhi_epi<52>(v_op1_5, v_op2_5);
    __m512i v_prod_hi_6 = _mm512_hexl_mulhi_epi<52>(v_op1_6, v_op2_6);
    __m512i v_prod_hi_7 = _mm512_hexl_mulhi_epi<52>(v_op1_7, v_op2_7);
    __m512i v_prod_hi_8 = _mm512_hexl_mulhi_epi<52>(v_op1_8, v_op2_8);
    __m512i v_prod_hi_9 = _mm512_hexl_mulhi_epi<52>(v_op1_9, v_op2_9);
    __m512i v_prod_hi_10 = _mm512_hexl_mulhi_epi<52>(v_op1_10, v_op2_10);
    __m512i v_prod_hi_11 = _mm512_hexl_mulhi_epi<52>(v_op1_11, v_op2_11);
    __m512i v_prod_hi_12 = _mm512_hexl_mulhi_epi<52>(v_op1_12, v_op2_12);
    __m512i v_prod_hi_13 = _mm512_hexl_mulhi_epi<52>(v_op1_13, v_op2_13);
    __m512i v_prod_hi_14 = _mm512_hexl_mulhi_epi<52>(v_op1_14, v_op2_14);
    __m512i v_prod_hi_15 = _mm512_hexl_mulhi_epi<52>(v_op1_15, v_op2_15);
    __m512i v_prod_hi_16 = _mm512_hexl_mulhi_epi<52>(v_op1_16, v_op2_16);

    __m512i v_prod_lo_1 = _mm512_hexl_mullo_epi<52>(v_op1_1, v_op2_1);
    __m512i v_prod_lo_2 = _mm512_hexl_mullo_epi<52>(v_op1_2, v_op2_2);
    __m512i v_prod_lo_3 = _mm512_hexl_mullo_epi<52>(v_op1_3, v_op2_3);
    __m512i v_prod_lo_4 = _mm512_hexl_mullo_epi<52>(v_op1_4, v_op2_4);
    __m512i v_prod_lo_5 = _mm512_hexl_mullo_epi<52>(v_op1_5, v_op2_5);
    __m512i v_prod_lo_6 = _mm512_hexl_mullo_epi<52>(v_op1_6, v_op2_6);
    __m512i v_prod_lo_7 = _mm512_hexl_mullo_epi<52>(v_op1_7, v_op2_7);
    __m512i v_prod_lo_8 = _mm512_hexl_mullo_epi<52>(v_op1_8, v_op2_8);
    __m512i v_prod_lo_9 = _mm512_hexl_mullo_epi<52>(v_op1_9, v_op2_9);
    __m512i v_prod_lo_10 = _mm512_hexl_mullo_epi<52>(v_op1_10, v_op2_10);
    __m512i v_prod_lo_11 = _mm512_hexl_mullo_epi<52>(v_op1_11, v_op2_11);
    __m512i v_prod_lo_12 = _mm512_hexl_mullo_epi<52>(v_op1_12, v_op2_12);
    __m512i v_prod_lo_13 = _mm512_hexl_mullo_epi<52>(v_op1_13, v_op2_13);
    __m512i v_prod_lo_14 = _mm512_hexl_mullo_epi<52>(v_op1_14, v_op2_14);
    __m512i v_prod_lo_15 = _mm512_hexl_mullo_epi<52>(v_op1_15, v_op2_15);
    __m512i v_prod_lo_16 = _mm512_hexl_mullo_epi<52>(v_op1_16, v_op2_16);

    __m512i c1_lo_1 = _mm512_srli_epi64(v_prod_lo_1, Nm1);
    __m512i c1_lo_2 = _mm512_srli_epi64(v_prod_lo_2, Nm1);
    __m512i c1_lo_3 = _mm512_srli_epi64(v_prod_lo_3, Nm1);
    __m512i c1_lo_4 = _mm512_srli_epi64(v_prod_lo_4, Nm1);
    __m512i c1_lo_5 = _mm512_srli_epi64(v_prod_lo_5, Nm1);
    __m512i c1_lo_6 = _mm512_srli_epi64(v_prod_lo_6, Nm1);
    __m512i c1_lo_7 = _mm512_srli_epi64(v_prod_lo_7, Nm1);
    __m512i c1_lo_8 = _mm512_srli_epi64(v_prod_lo_8, Nm1);
    __m512i c1_lo_9 = _mm512_srli_epi64(v_prod_lo_9, Nm1);
    __m512i c1_lo_10 = _mm512_srli_epi64(v_prod_lo_10, Nm1);
    __m512i c1_lo_11 = _mm512_srli_epi64(v_prod_lo_11, Nm1);
    __m512i c1_lo_12 = _mm512_srli_epi64(v_prod_lo_12, Nm1);
    __m512i c1_lo_13 = _mm512_srli_epi64(v_prod_lo_13, Nm1);
    __m512i c1_lo_14 = _mm512_srli_epi64(v_prod_lo_14, Nm1);
    __m512i c1_lo_15 = _mm512_srli_epi64(v_prod_lo_15, Nm1);
    __m512i c1_lo_16 = _mm512_srli_epi64(v_prod_lo_16, Nm1);

    __m512i c1_hi_1 = _mm512_slli_epi64(v_prod_hi_1, HiShift);
    __m512i c1_hi_2 = _mm512_slli_epi64(v_prod_hi_2, HiShift);
    __m512i c1_hi_3 = _mm512_slli_epi64(v_prod_hi_3, HiShift);
    __m512i c1_hi_4 = _mm512_slli_epi64(v_prod_hi_4, HiShift);
    __m512i c1_hi_5 = _mm512_slli_epi64(v_prod_hi_5, HiShift);
    __m512i c1_hi_6 = _mm512_slli_epi64(v_prod_hi_6, HiShift);
    __m512i c1_hi_7 = _mm512_slli_epi64(v_prod_hi_7, HiShift);
    __m512i c1_hi_8 = _mm512_slli_epi64(v_prod_hi_8, HiShift);
    __m512i c1_hi_9 = _mm512_slli_epi64(v_prod_hi_9, HiShift);
    __m512i c1_hi_10 = _mm512_slli_epi64(v_prod_hi_10, HiShift);
    __m512i c1_hi_11 = _mm512_slli_epi64(v_prod_hi_11, HiShift);
    __m512i c1_hi_12 = _mm512_slli_epi64(v_prod_hi_12, HiShift);
    __m512i c1_hi_13 = _mm512_slli_epi64(v_prod_hi_13, HiShift);
    __m512i c1_hi_14 = _mm512_slli_epi64(v_prod_hi_14, HiShift);
    __m512i c1_hi_15 = _mm512_slli_epi64(v_prod_hi_15, HiShift);
    __m512i c1_hi_16 = _mm512_slli_epi64(v_prod_hi_16, HiShift);

    __m512i c1_1 = _mm512_or_epi64(c1_lo_1, c1_hi_1);
    __m512i c1_2 = _mm512_or_epi64(c1_lo_2, c1_hi_2);
    __m512i c1_3 = _mm512_or_epi64(c1_lo_3, c1_hi_3);
    __m512i c1_4 = _mm512_or_epi64(c1_lo_4, c1_hi_4);
    __m512i c1_5 = _mm512_or_epi64(c1_lo_5, c1_hi_5);
    __m512i c1_6 = _mm512_or_epi64(c1_lo_6, c1_hi_6);
    __m512i c1_7 = _mm512_or_epi64(c1_lo_7, c1_hi_7);
    __m512i c1_8 = _mm512_or_epi64(c1_lo_8, c1_hi_8);
    __m512i c1_9 = _mm512_or_epi64(c1_lo_9, c1_hi_9);
    __m512i c1_10 = _mm512_or_epi64(c1_lo_10, c1_hi_10);
    __m512i c1_11 = _mm512_or_epi64(c1_lo_11, c1_hi_11);
    __m512i c1_12 = _mm512_or_epi64(c1_lo_12, c1_hi_12);
    __m512i c1_13 = _mm512_or_epi64(c1_lo_13, c1_hi_13);
    __m512i c1_14 = _mm512_or_epi64(c1_lo_14, c1_hi_14);
    __m512i c1_15 = _mm512_or_epi64(c1_lo_15, c1_hi_15);
    __m512i c1_16 = _mm512_or_epi64(c1_lo_16, c1_hi_16);

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
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_1, c3_1, v_neg_mod);
    __m512i c4_2 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_2, c3_2, v_neg_mod);
    __m512i c4_3 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_3, c3_3, v_neg_mod);
    __m512i c4_4 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_4, c3_4, v_neg_mod);
    __m512i c4_5 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_5, c3_5, v_neg_mod);
    __m512i c4_6 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_6, c3_6, v_neg_mod);
    __m512i c4_7 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_7, c3_7, v_neg_mod);
    __m512i c4_8 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_8, c3_8, v_neg_mod);
    __m512i c4_9 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_9, c3_9, v_neg_mod);
    __m512i c4_10 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_10, c3_10, v_neg_mod);
    __m512i c4_11 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_11, c3_11, v_neg_mod);
    __m512i c4_12 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_12, c3_12, v_neg_mod);
    __m512i c4_13 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_13, c3_13, v_neg_mod);
    __m512i c4_14 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_14, c3_14, v_neg_mod);
    __m512i c4_15 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_15, c3_15, v_neg_mod);
    __m512i c4_16 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_16, c3_16, v_neg_mod);

    // TODO(fboemer): remote v_twice-mod agruent?

    __m512i v_result_1 = _mm512_hexl_small_mod_epu64<2>(c4_1, v_modulus);
    __m512i v_result_2 = _mm512_hexl_small_mod_epu64<2>(c4_2, v_modulus);
    __m512i v_result_3 =
        _mm512_hexl_small_mod_epu64<2>(c4_3, v_modulus, &v_twice_mod);
    __m512i v_result_4 =
        _mm512_hexl_small_mod_epu64<2>(c4_4, v_modulus, &v_twice_mod);
    __m512i v_result_5 =
        _mm512_hexl_small_mod_epu64<2>(c4_5, v_modulus, &v_twice_mod);
    __m512i v_result_6 =
        _mm512_hexl_small_mod_epu64<2>(c4_6, v_modulus, &v_twice_mod);
    __m512i v_result_7 =
        _mm512_hexl_small_mod_epu64<2>(c4_7, v_modulus, &v_twice_mod);
    __m512i v_result_8 =
        _mm512_hexl_small_mod_epu64<2>(c4_8, v_modulus, &v_twice_mod);
    __m512i v_result_9 =
        _mm512_hexl_small_mod_epu64<2>(c4_9, v_modulus, &v_twice_mod);
    __m512i v_result_10 =
        _mm512_hexl_small_mod_epu64<2>(c4_10, v_modulus, &v_twice_mod);
    __m512i v_result_11 =
        _mm512_hexl_small_mod_epu64<2>(c4_11, v_modulus, &v_twice_mod);
    __m512i v_result_12 =
        _mm512_hexl_small_mod_epu64<2>(c4_12, v_modulus, &v_twice_mod);
    __m512i v_result_13 =
        _mm512_hexl_small_mod_epu64<2>(c4_13, v_modulus, &v_twice_mod);
    __m512i v_result_14 =
        _mm512_hexl_small_mod_epu64<2>(c4_14, v_modulus, &v_twice_mod);
    __m512i v_result_15 =
        _mm512_hexl_small_mod_epu64<2>(c4_15, v_modulus, &v_twice_mod);
    __m512i v_result_16 =
        _mm512_hexl_small_mod_epu64<2>(c4_16, v_modulus, &v_twice_mod);

    _mm512_storeu_si512(vp_result++, v_result_1);
    _mm512_storeu_si512(vp_result++, v_result_2);
    _mm512_storeu_si512(vp_result++, v_result_3);
    _mm512_storeu_si512(vp_result++, v_result_4);
    _mm512_storeu_si512(vp_result++, v_result_5);
    _mm512_storeu_si512(vp_result++, v_result_6);
    _mm512_storeu_si512(vp_result++, v_result_7);
    _mm512_storeu_si512(vp_result++, v_result_8);
    _mm512_storeu_si512(vp_result++, v_result_9);
    _mm512_storeu_si512(vp_result++, v_result_10);
    _mm512_storeu_si512(vp_result++, v_result_11);
    _mm512_storeu_si512(vp_result++, v_result_12);
    _mm512_storeu_si512(vp_result++, v_result_13);
    _mm512_storeu_si512(vp_result++, v_result_14);
    _mm512_storeu_si512(vp_result++, v_result_15);
    _mm512_storeu_si512(vp_result++, v_result_16);
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
    __m512i v_op1 = _mm512_loadu_si512(vp_operand1);
    v_op1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1, v_modulus,
                                                        &v_twice_mod);

    __m512i v_op2 = _mm512_loadu_si512(vp_operand2);
    v_op2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2, v_modulus,
                                                        &v_twice_mod);

    // Compute product
    __m512i v_prod_hi = _mm512_hexl_mulhi_epi<52>(v_op1, v_op2);
    __m512i v_prod_lo = _mm512_hexl_mullo_epi<52>(v_op1, v_op2);

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
  unsigned int Nm1 = static_cast<unsigned int>(N - 1);
  unsigned int HiShift = static_cast<unsigned int>(53 - N);

  HEXL_UNUSED(v_twice_mod);
  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op1 = _mm512_loadu_si512(vp_operand1);
    v_op1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1, v_modulus,
                                                        &v_twice_mod);

    __m512i v_op2 = _mm512_loadu_si512(vp_operand2);
    v_op2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2, v_modulus,
                                                        &v_twice_mod);

    // Compute product
    __m512i v_prod_hi = _mm512_hexl_mulhi_epi<52>(v_op1, v_op2);
    __m512i v_prod_lo = _mm512_hexl_mullo_epi<52>(v_op1, v_op2);

    __m512i c1_lo = _mm512_srli_epi64(v_prod_lo, Nm1);
    __m512i c1_hi = _mm512_slli_epi64(v_prod_hi, HiShift);
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
  switch (n) {
    case 1024:
      EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor, 1024>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;

    case 2048:
      EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor, 2048>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;

    case 4096:
      EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor, 4096>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;

    case 8192:
      EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor, 8192>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;

    case 16384:
      EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor, 16384>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;

    case 32768:
      EltwiseMultModAVX512IFMAIntLoopUnroll<BitShift, InputModFactor, 32768>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;

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
