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

template <int ProdRightShift, int InputModFactor, int CoeffCount>
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

  constexpr unsigned int HiShift =
      static_cast<unsigned int>(52 - ProdRightShift);

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

    __m512i c1_lo_1 = _mm512_srli_epi64(v_prod_lo_1, ProdRightShift);
    __m512i c1_lo_2 = _mm512_srli_epi64(v_prod_lo_2, ProdRightShift);
    __m512i c1_lo_3 = _mm512_srli_epi64(v_prod_lo_3, ProdRightShift);
    __m512i c1_lo_4 = _mm512_srli_epi64(v_prod_lo_4, ProdRightShift);
    __m512i c1_lo_5 = _mm512_srli_epi64(v_prod_lo_5, ProdRightShift);
    __m512i c1_lo_6 = _mm512_srli_epi64(v_prod_lo_6, ProdRightShift);
    __m512i c1_lo_7 = _mm512_srli_epi64(v_prod_lo_7, ProdRightShift);
    __m512i c1_lo_8 = _mm512_srli_epi64(v_prod_lo_8, ProdRightShift);
    __m512i c1_lo_9 = _mm512_srli_epi64(v_prod_lo_9, ProdRightShift);
    __m512i c1_lo_10 = _mm512_srli_epi64(v_prod_lo_10, ProdRightShift);
    __m512i c1_lo_11 = _mm512_srli_epi64(v_prod_lo_11, ProdRightShift);
    __m512i c1_lo_12 = _mm512_srli_epi64(v_prod_lo_12, ProdRightShift);
    __m512i c1_lo_13 = _mm512_srli_epi64(v_prod_lo_13, ProdRightShift);
    __m512i c1_lo_14 = _mm512_srli_epi64(v_prod_lo_14, ProdRightShift);
    __m512i c1_lo_15 = _mm512_srli_epi64(v_prod_lo_15, ProdRightShift);
    __m512i c1_lo_16 = _mm512_srli_epi64(v_prod_lo_16, ProdRightShift);

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

    __m512i q_hat_1 = _mm512_hexl_mulhi_epi<52>(c1_1, v_barr_lo);
    __m512i q_hat_2 = _mm512_hexl_mulhi_epi<52>(c1_2, v_barr_lo);
    __m512i q_hat_3 = _mm512_hexl_mulhi_epi<52>(c1_3, v_barr_lo);
    __m512i q_hat_4 = _mm512_hexl_mulhi_epi<52>(c1_4, v_barr_lo);
    __m512i q_hat_5 = _mm512_hexl_mulhi_epi<52>(c1_5, v_barr_lo);
    __m512i q_hat_6 = _mm512_hexl_mulhi_epi<52>(c1_6, v_barr_lo);
    __m512i q_hat_7 = _mm512_hexl_mulhi_epi<52>(c1_7, v_barr_lo);
    __m512i q_hat_8 = _mm512_hexl_mulhi_epi<52>(c1_8, v_barr_lo);
    __m512i q_hat_9 = _mm512_hexl_mulhi_epi<52>(c1_9, v_barr_lo);
    __m512i q_hat_10 = _mm512_hexl_mulhi_epi<52>(c1_10, v_barr_lo);
    __m512i q_hat_11 = _mm512_hexl_mulhi_epi<52>(c1_11, v_barr_lo);
    __m512i q_hat_12 = _mm512_hexl_mulhi_epi<52>(c1_12, v_barr_lo);
    __m512i q_hat_13 = _mm512_hexl_mulhi_epi<52>(c1_13, v_barr_lo);
    __m512i q_hat_14 = _mm512_hexl_mulhi_epi<52>(c1_14, v_barr_lo);
    __m512i q_hat_15 = _mm512_hexl_mulhi_epi<52>(c1_15, v_barr_lo);
    __m512i q_hat_16 = _mm512_hexl_mulhi_epi<52>(c1_16, v_barr_lo);

    __m512i z_1 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_1, q_hat_1, v_neg_mod);
    __m512i z_2 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_2, q_hat_2, v_neg_mod);
    __m512i z_3 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_3, q_hat_3, v_neg_mod);
    __m512i z_4 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_4, q_hat_4, v_neg_mod);
    __m512i z_5 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_5, q_hat_5, v_neg_mod);
    __m512i z_6 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_6, q_hat_6, v_neg_mod);
    __m512i z_7 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_7, q_hat_7, v_neg_mod);
    __m512i z_8 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_8, q_hat_8, v_neg_mod);
    __m512i z_9 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_9, q_hat_9, v_neg_mod);
    __m512i z_10 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_10, q_hat_10, v_neg_mod);
    __m512i z_11 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_11, q_hat_11, v_neg_mod);
    __m512i z_12 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_12, q_hat_12, v_neg_mod);
    __m512i z_13 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_13, q_hat_13, v_neg_mod);
    __m512i z_14 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_14, q_hat_14, v_neg_mod);
    __m512i z_15 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_15, q_hat_15, v_neg_mod);
    __m512i z_16 =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo_16, q_hat_16, v_neg_mod);

    __m512i v_result_1 = _mm512_hexl_small_mod_epu64<2>(z_1, v_modulus);
    __m512i v_result_2 = _mm512_hexl_small_mod_epu64<2>(z_2, v_modulus);
    __m512i v_result_3 = _mm512_hexl_small_mod_epu64<2>(z_3, v_modulus);
    __m512i v_result_4 = _mm512_hexl_small_mod_epu64<2>(z_4, v_modulus);
    __m512i v_result_5 = _mm512_hexl_small_mod_epu64<2>(z_5, v_modulus);
    __m512i v_result_6 = _mm512_hexl_small_mod_epu64<2>(z_6, v_modulus);
    __m512i v_result_7 = _mm512_hexl_small_mod_epu64<2>(z_7, v_modulus);
    __m512i v_result_8 = _mm512_hexl_small_mod_epu64<2>(z_8, v_modulus);
    __m512i v_result_9 = _mm512_hexl_small_mod_epu64<2>(z_9, v_modulus);
    __m512i v_result_10 = _mm512_hexl_small_mod_epu64<2>(z_10, v_modulus);
    __m512i v_result_11 = _mm512_hexl_small_mod_epu64<2>(z_11, v_modulus);
    __m512i v_result_12 = _mm512_hexl_small_mod_epu64<2>(z_12, v_modulus);
    __m512i v_result_13 = _mm512_hexl_small_mod_epu64<2>(z_13, v_modulus);
    __m512i v_result_14 = _mm512_hexl_small_mod_epu64<2>(z_14, v_modulus);
    __m512i v_result_15 = _mm512_hexl_small_mod_epu64<2>(z_15, v_modulus);
    __m512i v_result_16 = _mm512_hexl_small_mod_epu64<2>(z_16, v_modulus);

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

// Algorithm 2 from https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf
template <int ProdRightShift, int InputModFactor>
void EltwiseMultModAVX512IFMAIntLoopDefault(
    __m512i* vp_result, const __m512i* vp_operand1, const __m512i* vp_operand2,
    __m512i v_barr_lo, __m512i v_modulus, __m512i v_neg_mod,
    __m512i v_twice_mod, uint64_t n) {
  HEXL_UNUSED(v_twice_mod);
  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_op1 = _mm512_loadu_si512(vp_operand1);
    v_op1 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op1, v_modulus,
                                                        &v_twice_mod);

    __m512i v_op2 = _mm512_loadu_si512(vp_operand2);
    v_op2 = _mm512_hexl_small_mod_epu64<InputModFactor>(v_op2, v_modulus,
                                                        &v_twice_mod);

    // Compute product U
    __m512i v_prod_hi = _mm512_hexl_mulhi_epi<52>(v_op1, v_op2);
    __m512i v_prod_lo = _mm512_hexl_mullo_epi<52>(v_op1, v_op2);

    // c1 = floor(U / 2^{n + beta})
    __m512i c1_lo =
        _mm512_srli_epi64(v_prod_lo, static_cast<unsigned int>(ProdRightShift));
    __m512i c1_hi = _mm512_slli_epi64(
        v_prod_hi, static_cast<unsigned int>(52ULL - (ProdRightShift)));
    __m512i c1 = _mm512_or_epi64(c1_lo, c1_hi);

    // alpha - beta == 52, so we only need high 52 bits
    __m512i q_hat = _mm512_hexl_mulhi_epi<52>(c1, v_barr_lo);

    // Z = prod_lo - (p * q_hat)_lo
    __m512i v_result =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo, q_hat, v_neg_mod);

    // Reduce result to [0, q)
    v_result = _mm512_hexl_small_mod_epu64<2>(v_result, v_modulus);
    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

// Algorithm 2 from https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf
template <int InputModFactor>
void EltwiseMultModAVX512IFMAIntLoopDefault(
    __m512i* vp_result, const __m512i* vp_operand1, const __m512i* vp_operand2,
    __m512i v_barr_lo, __m512i v_modulus, __m512i v_neg_mod,
    __m512i v_twice_mod, uint64_t n, uint64_t prod_right_shift) {
  unsigned int low_shift = static_cast<unsigned int>(prod_right_shift);
  unsigned int high_shift = static_cast<unsigned int>(52 - prod_right_shift);

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

    __m512i c1_lo = _mm512_srli_epi64(v_prod_lo, low_shift);
    __m512i c1_hi = _mm512_slli_epi64(v_prod_hi, high_shift);
    __m512i c1 = _mm512_or_epi64(c1_lo, c1_hi);

    // alpha - beta == 52, so we only need high 52 bits
    __m512i q_hat = _mm512_hexl_mulhi_epi<52>(c1, v_barr_lo);

    // z = prod_lo - (p * q_hat)_lo
    __m512i v_result =
        _mm512_hexl_mullo_add_lo_epi<52>(v_prod_lo, q_hat, v_neg_mod);

    // Reduce result to [0, q)
    v_result = _mm512_hexl_small_mod_epu64<2>(v_result, v_modulus);

    _mm512_storeu_si512(vp_result, v_result);

    ++vp_operand1;
    ++vp_operand2;
    ++vp_result;
  }
}

template <int ProdRightShift, int InputModFactor>
void EltwiseMultModAVX512IFMAIntLoop(__m512i* vp_result,
                                     const __m512i* vp_operand1,
                                     const __m512i* vp_operand2,
                                     __m512i v_barr_lo, __m512i v_modulus,
                                     __m512i v_neg_mod, __m512i v_twice_mod,
                                     uint64_t n) {
  switch (n) {
    case 1024: {
      EltwiseMultModAVX512IFMAIntLoopUnroll<ProdRightShift, InputModFactor,
                                            1024>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;
    }
    case 2048: {
      EltwiseMultModAVX512IFMAIntLoopUnroll<ProdRightShift, InputModFactor,
                                            2048>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;
    }
    case 4096: {
      EltwiseMultModAVX512IFMAIntLoopUnroll<ProdRightShift, InputModFactor,
                                            4096>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;
    }
    case 8192: {
      EltwiseMultModAVX512IFMAIntLoopUnroll<ProdRightShift, InputModFactor,
                                            8192>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;
    }
    case 16384: {
      EltwiseMultModAVX512IFMAIntLoopUnroll<ProdRightShift, InputModFactor,
                                            16384>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;
    }
    case 32768: {
      EltwiseMultModAVX512IFMAIntLoopUnroll<ProdRightShift, InputModFactor,
                                            32768>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod);
      break;
    }
    default:
      EltwiseMultModAVX512IFMAIntLoopDefault<ProdRightShift, InputModFactor>(
          vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,
          v_twice_mod, n);
  }
}

#define ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(ProdRightShift, \
                                                               InputModFactor) \
  case (ProdRightShift): {                                                     \
    EltwiseMultModAVX512IFMAIntLoop<(ProdRightShift), (InputModFactor)>(       \
        vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus, v_neg_mod,  \
        v_twice_mod, n);                                                       \
    break;                                                                     \
  }

// Algorithm 2 from https://homes.esat.kuleuven.be/~fvercaut/papers/bar_mont.pdf
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

  constexpr int64_t beta = -2;
  HEXL_CHECK(beta <= -2, "beta must be <= -2 for correctness");
  constexpr int64_t alpha = 50;  // ensures alpha - beta = 52
  uint64_t gamma = Log2(InputModFactor);
  HEXL_UNUSED(gamma);
  HEXL_CHECK(alpha >= gamma + 1, "alpha must be >= gamma + 1 for correctness");

  const uint64_t ceil_log_mod = Log2(modulus) + 1;  // "n" from Algorithm 2
  uint64_t prod_right_shift = ceil_log_mod + beta;

  // Barrett factor "mu"
  // TODO(fboemer): Allow MultiplyFactor to take bit shifts != 52
  HEXL_CHECK(ceil_log_mod + alpha >= 52, "ceil_log_mod + alpha < 52");
  uint64_t barr_lo =
      MultiplyFactor((1ULL << (ceil_log_mod + alpha - 52)), 52, modulus)
          .BarrettFactor();

  __m512i v_barr_lo = _mm512_set1_epi64(static_cast<int64_t>(barr_lo));
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(2 * modulus));
  __m512i v_neg_mod = _mm512_set1_epi64(-static_cast<int64_t>(modulus));
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  // Let d be the product operand1 * operand2.
  // To ensure d >> prod_right_shift < (1ULL << 52), we need
  // (input_mod_factor * modulus)^2 >> (prod_right_shift) < (1ULL << 52)
  // This happens when 2*log_2(input_mod_factor) + ceil_log_mod - beta < 51
  // If not, we need to reduce the inputs to be less than modulus for
  // correctness. This is less efficient, so we avoid it when possible.
  bool reduce_mod = 2 * Log2(InputModFactor) + prod_right_shift - beta >= 51;

  if (reduce_mod) {
    // Here, we assume beta = -2
    HEXL_CHECK(beta == -2, "beta != -2 may skip some cases");
    // This reduce_mod case happens only when
    // prod_right_shift >= 51 - 2 * log2(input_mod_factor) >= 45.
    // Additionally, modulus < (1ULL << 50) implies
    // prod_right_shift <= 49. So N == 45, 46, 47, 48, 49 are the
    // only cases here.
    switch (prod_right_shift) {
      // The template arguments are required for use of _mm512_hexl_shrdi_epi64,
      // which requires a compile-time constant for the shift.
      ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(45, InputModFactor)
      ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(46, InputModFactor)
      ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(47, InputModFactor)
      ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(48, InputModFactor)
      ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(49, InputModFactor)
      default: {
        HEXL_CHECK(false,
                   "Bad value for prod_right_shift: " << prod_right_shift);
      }
    }
  } else {
    switch (prod_right_shift) {
      // Smaller shifts are uncommon.
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(15, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(16, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(17, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(18, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(19, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(20, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(21, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(22, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(23, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(24, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(25, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(26, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(27, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(28, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(29, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(31, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(32, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(33, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(34, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(35, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(36, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(37, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(38, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(39, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(40, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(41, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(42, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(43, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(44, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(45, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(46, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(47, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(48, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(49, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(50, 1)
      // ELTWISE_MULT_MOD_AVX512_IFMA_INT_PROD_RIGHT_SHIFT_CASE(51, 1)
      default: {
        EltwiseMultModAVX512IFMAIntLoopDefault<1>(
            vp_result, vp_operand1, vp_operand2, v_barr_lo, v_modulus,
            v_neg_mod, v_twice_mod, n, prod_right_shift);
      }
    }
  }
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

#endif

}  // namespace hexl
}  // namespace intel
