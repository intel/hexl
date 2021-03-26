// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>
#include <stdint.h>

#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "intel-hexl/eltwise/eltwise-fma-mod.hpp"
#include "number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"
#include "util/check.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

template <int BitShift, int InputModFactor>
void EltwiseFMAModAVX512(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                         const uint64_t* arg3, uint64_t n, uint64_t modulus) {
  HEXL_CHECK(modulus < MaximumValue(BitShift),
             "Modulus " << modulus << " exceeds bit shift bound "
                        << MaximumValue(BitShift));
  HEXL_CHECK(modulus != 0, "Require modulus != 0");

  HEXL_CHECK(arg1, "arg1 == nullptr");
  HEXL_CHECK(result, "result == nullptr");

  HEXL_CHECK_BOUNDS(arg1, n, InputModFactor * modulus,
                    "arg1 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK_BOUNDS(&arg2, 1, InputModFactor * modulus,
                    "arg2 exceeds bound " << (InputModFactor * modulus));
  HEXL_CHECK(BitShift == 52 || BitShift == 64,
             "Invalid bitshift " << BitShift << "; need 52 or 64");

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseFMAModNative<InputModFactor>(result, arg1, arg2, arg3, n_mod_8,
                                        modulus);
    arg1 += n_mod_8;
    if (arg3 != nullptr) {
      arg3 += n_mod_8;
    }
    result += n_mod_8;
    n -= n_mod_8;
  }

  uint64_t twice_modulus = 2 * modulus;
  uint64_t four_times_modulus = 4 * modulus;
  arg2 = ReduceMod<InputModFactor>(arg2, modulus, &twice_modulus,
                                   &four_times_modulus);
  uint64_t arg2_barr = MultiplyFactor(arg2, BitShift, modulus).BarrettFactor();

  __m512i varg2_barr = _mm512_set1_epi64(arg2_barr);

  __m512i vmodulus = _mm512_set1_epi64(modulus);
  __m512i v2_modulus = _mm512_set1_epi64(2 * modulus);
  __m512i v4_modulus = _mm512_set1_epi64(4 * modulus);
  const __m512i* vp_arg1 = reinterpret_cast<const __m512i*>(arg1);
  __m512i varg2 = _mm512_set1_epi64(arg2);
  varg2 = _mm512_hexl_small_mod_epu64<InputModFactor>(varg2, vmodulus,
                                                      &v2_modulus, &v4_modulus);

  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  if (arg3) {
    const __m512i* vp_arg3 = reinterpret_cast<const __m512i*>(arg3);
    HEXL_LOOP_UNROLL_4
    for (size_t i = n / 8; i > 0; --i) {
      __m512i varg1 = _mm512_loadu_si512(vp_arg1);
      __m512i varg3 = _mm512_loadu_si512(vp_arg3);

      varg1 = _mm512_hexl_small_mod_epu64<InputModFactor>(
          varg1, vmodulus, &v2_modulus, &v4_modulus);
      varg3 = _mm512_hexl_small_mod_epu64<InputModFactor>(
          varg3, vmodulus, &v2_modulus, &v4_modulus);

      __m512i va_times_b = _mm512_hexl_mullo_epi<64>(varg1, varg2);

      __m512i vq = _mm512_hexl_mulhi_epi<BitShift>(varg1, varg2_barr);
      __m512i vq_times_mod = _mm512_mullo_epi64(vq, vmodulus);
      vq = _mm512_sub_epi64(va_times_b, vq_times_mod);
      // Conditional Barrett subtraction
      vq = _mm512_hexl_small_mod_epu64(vq, vmodulus);

      vq = _mm512_add_epi64(vq, varg3);
      vq = _mm512_hexl_small_mod_epu64(vq, vmodulus);

      _mm512_storeu_si512(vp_result, vq);

      ++vp_arg1;
      ++vp_result;
      ++vp_arg3;
    }
  } else {  // arg3 == nullptr
    HEXL_LOOP_UNROLL_4
    for (size_t i = n / 8; i > 0; --i) {
      __m512i varg1 = _mm512_loadu_si512(vp_arg1);
      varg1 = _mm512_hexl_small_mod_epu64<InputModFactor>(
          varg1, vmodulus, &v2_modulus, &v4_modulus);

      __m512i vq = _mm512_hexl_mulhi_epi<BitShift>(varg1, varg2_barr);
      __m512i vq_times_mod = _mm512_mullo_epi64(vq, vmodulus);
      __m512i va_times_b = _mm512_hexl_mullo_epi<64>(varg1, varg2);
      vq = _mm512_sub_epi64(va_times_b, vq_times_mod);
      // Conditional Barrett subtraction
      vq = _mm512_hexl_small_mod_epu64(vq, vmodulus);
      _mm512_storeu_si512(vp_result, vq);

      ++vp_arg1;
      ++vp_result;
    }
  }
}

#endif  //  HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
