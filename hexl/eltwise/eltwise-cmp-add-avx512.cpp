// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-cmp-add-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/util.hpp"
#include "thread-pool/thread-pool-executor.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
void EltwiseCmpAddAVX512(uint64_t* result, const uint64_t* operand1, uint64_t n,
                         CMPINT cmp, uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(diff != 0, "Require diff != 0");

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseCmpAddNative(result, operand1, n_mod_8, cmp, bound, diff);
    operand1 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_bound = _mm512_set1_epi64(static_cast<int64_t>(bound));
  const __m512i* vp_op = reinterpret_cast<const __m512i*>(operand1);
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);

  ThreadPoolExecutor::AddParallelJobs(
      n / 8, [vp_result, vp_op, v_bound, cmp, diff](size_t start, size_t end) {
        auto in_vp_result = vp_result + start;
        auto in_vp_op = vp_op + start;

        for (size_t i = start; i < end; ++i) {
          __m512i v_op = _mm512_loadu_si512(in_vp_op);
          __m512i v_add_diff = _mm512_hexl_cmp_epi64(v_op, v_bound, cmp, diff);
          v_op = _mm512_add_epi64(v_op, v_add_diff);
          _mm512_storeu_si512(in_vp_result, v_op);

          ++in_vp_result;
          ++in_vp_op;
        }
      });
}
#endif

}  // namespace hexl
}  // namespace intel
