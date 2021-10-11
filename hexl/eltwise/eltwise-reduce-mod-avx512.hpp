// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <vector>

#include "eltwise/eltwise-reduce-mod-avx512.hpp"
#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
template <int BitShift = 64>
void EltwiseReduceModAVX512(uint64_t* result, const uint64_t* operand,
                            uint64_t n, uint64_t modulus,
                            uint64_t input_mod_factor,
                            uint64_t output_mod_factor) {
  HEXL_CHECK(operand != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(input_mod_factor == modulus || input_mod_factor == 2 ||
                 input_mod_factor == 4,
             "input_mod_factor must be modulus or 2 or 4" << input_mod_factor);
  HEXL_CHECK(output_mod_factor == 1 || output_mod_factor == 2,
             "output_mod_factor must be 1 or 2 " << output_mod_factor);
  HEXL_CHECK(input_mod_factor != output_mod_factor,
             "input_mod_factor must not be equal to output_mod_factor ");

  uint64_t n_tmp = n;

  // Multi-word Barrett reduction precomputation
  constexpr int64_t alpha = BitShift - 2;
  constexpr int64_t beta = -2;
  const uint64_t ceil_log_mod = Log2(modulus) + 1;  // "n" from Algorithm 2
  uint64_t prod_right_shift = ceil_log_mod + beta;
  __m512i v_neg_mod = _mm512_set1_epi64(-static_cast<int64_t>(modulus));

  uint64_t barrett_factor =
      MultiplyFactor(uint64_t(1) << (ceil_log_mod + alpha - BitShift), BitShift,
                     modulus)
          .BarrettFactor();

  uint64_t barrett_factor_52 = MultiplyFactor(1, 52, modulus).BarrettFactor();

  if (BitShift == 64) {
    // Single-worded Barrett reduction.
    barrett_factor = MultiplyFactor(1, 64, modulus).BarrettFactor();
  }

  __m512i v_bf = _mm512_set1_epi64(static_cast<int64_t>(barrett_factor));
  __m512i v_bf_52 = _mm512_set1_epi64(static_cast<int64_t>(barrett_factor_52));

  // Deals with n not divisible by 8
  uint64_t n_mod_8 = n_tmp % 8;
  if (n_mod_8 != 0) {
    EltwiseReduceModNative(result, operand, n_mod_8, modulus, input_mod_factor,
                           output_mod_factor);
    operand += n_mod_8;
    result += n_mod_8;
    n_tmp -= n_mod_8;
  }

  uint64_t twice_mod = modulus << 1;
  const __m512i* v_operand = reinterpret_cast<const __m512i*>(operand);
  __m512i* v_result = reinterpret_cast<__m512i*>(result);
  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i v_twice_mod = _mm512_set1_epi64(static_cast<int64_t>(twice_mod));

  if (input_mod_factor == modulus) {
    if (output_mod_factor == 2) {
      for (size_t i = 0; i < n_tmp; i += 8) {
        __m512i v_op = _mm512_loadu_si512(v_operand);
        v_op = _mm512_hexl_barrett_reduce64<BitShift, 2>(
            v_op, v_modulus, v_bf, v_bf_52, prod_right_shift, v_neg_mod);
        HEXL_CHECK_BOUNDS(ExtractValues(v_op).data(), 8, modulus,
                          "v_op exceeds bound " << modulus);
        _mm512_storeu_si512(v_result, v_op);
        ++v_operand;
        ++v_result;
      }
    } else {
      for (size_t i = 0; i < n_tmp; i += 8) {
        __m512i v_op = _mm512_loadu_si512(v_operand);
        v_op = _mm512_hexl_barrett_reduce64<BitShift, 1>(
            v_op, v_modulus, v_bf, v_bf_52, prod_right_shift, v_neg_mod);
        HEXL_CHECK_BOUNDS(ExtractValues(v_op).data(), 8, modulus,
                          "v_op exceeds bound " << modulus);
        _mm512_storeu_si512(v_result, v_op);
        ++v_operand;
        ++v_result;
      }
    }
  }

  if (input_mod_factor == 2) {
    for (size_t i = 0; i < n_tmp; i += 8) {
      __m512i v_op = _mm512_loadu_si512(v_operand);
      v_op = _mm512_hexl_small_mod_epu64(v_op, v_modulus);
      HEXL_CHECK_BOUNDS(ExtractValues(v_op).data(), 8, modulus,
                        "v_op exceeds bound " << modulus);
      _mm512_storeu_si512(v_result, v_op);
      ++v_operand;
      ++v_result;
    }
  }

  if (input_mod_factor == 4) {
    if (output_mod_factor == 1) {
      for (size_t i = 0; i < n_tmp; i += 8) {
        __m512i v_op = _mm512_loadu_si512(v_operand);
        v_op = _mm512_hexl_small_mod_epu64(v_op, v_twice_mod);
        v_op = _mm512_hexl_small_mod_epu64(v_op, v_modulus);
        HEXL_CHECK_BOUNDS(ExtractValues(v_op).data(), 8, modulus,
                          "v_op exceeds bound " << modulus);
        _mm512_storeu_si512(v_result, v_op);
        ++v_operand;
        ++v_result;
      }
    }
    if (output_mod_factor == 2) {
      for (size_t i = 0; i < n_tmp; i += 8) {
        __m512i v_op = _mm512_loadu_si512(v_operand);
        v_op = _mm512_hexl_small_mod_epu64(v_op, v_twice_mod);
        HEXL_CHECK_BOUNDS(ExtractValues(v_op).data(), 8, twice_mod,
                          "v_op exceeds bound " << twice_mod);
        _mm512_storeu_si512(v_result, v_op);
        ++v_operand;
        ++v_result;
      }
    }
  }
}

#endif

}  // namespace hexl
}  // namespace intel
