// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-sub-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include "eltwise/eltwise-sub-mod-internal.hpp"
#include "hexl/eltwise/eltwise-sub-mod.hpp"
#include "hexl/util/check.hpp"
#include "thread-pool/thread-pool-executor.hpp"
#include "util/avx512-util.hpp"

#ifdef HEXL_HAS_AVX512DQ

namespace intel {
namespace hexl {

void EltwiseSubModAVX512(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-sub value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2, n, modulus,
                    "pre-sub value in operand2 exceeds bound " << modulus);

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseSubModNative(result, operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

  ThreadPoolExecutor::AddParallelJobs(
      n / 8, [vp_result, vp_operand1, vp_operand2, v_modulus](size_t start,
                                                              size_t end) {
        auto in_vp_result = vp_result + start;
        auto in_vp_operand1 = vp_operand1 + start;
        auto in_vp_operand2 = vp_operand2 + start;

        HEXL_LOOP_UNROLL_4
        for (size_t i = start; i < end; ++i) {
          __m512i v_operand1 = _mm512_loadu_si512(in_vp_operand1);
          __m512i v_operand2 = _mm512_loadu_si512(in_vp_operand2);

          __m512i v_result = _mm512_hexl_small_sub_mod_epi64(
              v_operand1, v_operand2, v_modulus);

          _mm512_storeu_si512(in_vp_result, v_result);

          ++in_vp_result;
          ++in_vp_operand1;
          ++in_vp_operand2;
        }
      });
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

void EltwiseSubModAVX512_loop(__m512i* vp_result, const __m512i* vp_operand1,
                              __m512i v_operand2, __m512i v_modulus,
                              size_t start, size_t end) {
  HEXL_LOOP_UNROLL_4
  for (size_t i = start; i < end; ++i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);

    __m512i v_result =
        _mm512_hexl_small_sub_mod_epi64(v_operand1, v_operand2, v_modulus);

    _mm512_storeu_si512(vp_result, v_result);

    ++vp_result;
    ++vp_operand1;
  }
}

void EltwiseSubModAVX512(uint64_t* result, const uint64_t* operand1,
                         uint64_t operand2, uint64_t n, uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-sub value in operand1 exceeds bound " << modulus);
  HEXL_CHECK(operand2 < modulus, "Require operand2 < modulus");

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseSubModNative(result, operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  __m512i v_operand2 = _mm512_set1_epi64(static_cast<int64_t>(operand2));

  ThreadPoolExecutor::AddParallelJobs(
      n / 8, [vp_result, vp_operand1, v_operand2, v_modulus](size_t start,
                                                             size_t end) {
        EltwiseSubModAVX512_loop(vp_result + start, vp_operand1 + start,
                                 v_operand2, v_modulus, start, end);
      });

  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

}  // namespace hexl
}  // namespace intel

#endif
