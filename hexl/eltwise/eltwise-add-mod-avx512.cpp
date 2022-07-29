// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-add-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include <iostream>

#include "eltwise/eltwise-add-mod-internal.hpp"
#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/util/check.hpp"
#include "util/avx512-util.hpp"

#ifdef HEXL_HAS_AVX512DQ

namespace intel {
namespace hexl {

void EltwiseAddModAVX512(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-add value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2, n, modulus,
                    "pre-add value in operand2 exceeds bound " << modulus);

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseAddModNative(result, operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

  HEXL_LOOP_UNROLL_4
  for (size_t i = n / 8; i > 0; --i) {
    __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);
    __m512i v_operand2 = _mm512_loadu_si512(vp_operand2);

    __m512i v_result =
        _mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);

    _mm512_storeu_si512(vp_result, v_result);

    ++vp_result;
    ++vp_operand1;
    ++vp_operand2;
  }

  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

void EltwiseAddModAVX512_OMP(uint64_t* result, const uint64_t* operand1,
                             const uint64_t* operand2, uint64_t n,
                             uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-add value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2, n, modulus,
                    "pre-add value in operand2 exceeds bound " << modulus);

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseAddModNative(result, operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

  // std::cout << "ROCHA " << std::endl;
  // std::cout << "n " << n << std::endl;
  omp_set_num_threads(6);
#pragma omp parallel firstprivate(vp_operand1, vp_operand2, vp_result)
  {
    int id = omp_get_thread_num();
    int threads = omp_get_num_threads();
    __m512i* i_vp_result = vp_result + id * n / 8 / threads;
    const __m512i* i_vp_operand1 = vp_operand1 + id * n / 8 / threads;
    const __m512i* i_vp_operand2 = vp_operand2 + id * n / 8 / threads;
    HEXL_LOOP_UNROLL_4
    for (size_t i = n / 8 / threads; i > 0; --i) {
      __m512i v_operand1 = _mm512_loadu_si512(i_vp_operand1);
      __m512i v_operand2 = _mm512_loadu_si512(i_vp_operand2);

      __m512i v_result =
          _mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);

      _mm512_storeu_si512(i_vp_result, v_result);

      ++i_vp_result;
      ++i_vp_operand1;
      ++i_vp_operand2;
    }
  }

  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

static const size_t N = 23;

class ParallelEltwiseAddModAVX512 {
  __m512i* vp_result;
  const __m512i* vp_operand1;
  const __m512i* vp_operand2;
  __m512i v_modulus;

 public:
  void operator()(const tbb::blocked_range<size_t>& r) const {
    __m512i* i_vp_result = vp_result + r.begin();
    const __m512i* i_vp_operand1 = vp_operand1 + r.begin();
    const __m512i* i_vp_operand2 = vp_operand2 + r.begin();
    for (size_t i = r.begin(); i != r.end(); ++i) {
      __m512i v_operand1 = _mm512_loadu_si512(i_vp_operand1);
      __m512i v_operand2 = _mm512_loadu_si512(i_vp_operand2);

      __m512i v_result =
          _mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);

      _mm512_storeu_si512(i_vp_result, v_result);
      ++i_vp_result;
      ++i_vp_operand1;
      ++i_vp_operand2;
    }
  }
  ParallelEltwiseAddModAVX512(__m512i* vp_r, const __m512i* vp_op1,
                              const __m512i* vp_op2, __m512i v_m)
      : vp_result(vp_r),
        vp_operand1(vp_op1),
        vp_operand2(vp_op2),
        v_modulus(v_m) {}
};

void EltwiseAddModAVX512_TBB(uint64_t* result, const uint64_t* operand1,
                             const uint64_t* operand2, uint64_t n,
                             uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(operand2 != nullptr, "Require operand2 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-add value in operand1 exceeds bound " << modulus);
  HEXL_CHECK_BOUNDS(operand2, n, modulus,
                    "pre-add value in operand2 exceeds bound " << modulus);

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseAddModNative(result, operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    operand2 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i* vp_operand2 = reinterpret_cast<const __m512i*>(operand2);

  // std::cout << "ROCHA " << std::endl;
  // std::cout << "n " << n << std::endl;
  oneapi::tbb::task_arena arena(6);
  arena.execute([&] {
    // std::cout << this_task_arena::max_concurrency() << std::endl;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, n / 8),
        /*[&](const blocked_range<size_t>& r) {
            __m512i* i_vp_result = vp_result + r.begin();
            const __m512i* i_vp_operand1 = vp_operand1 + r.begin();
            const __m512i* i_vp_operand2 = vp_operand2 + r.begin();
          for ( size_t i = r.begin(); i != r.end(); ++i ) {
            __m512i v_operand1 = _mm512_loadu_si512(i_vp_operand1);
            __m512i v_operand2 = _mm512_loadu_si512(i_vp_operand2);

            __m512i v_result =
                _mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2,
        v_modulus);

            _mm512_storeu_si512(i_vp_result, v_result);
            ++i_vp_result;
            ++i_vp_operand1;
            ++i_vp_operand2;
          }
        }*/
        ParallelEltwiseAddModAVX512(vp_result, vp_operand1, vp_operand2,
                                    v_modulus),
        tbb::static_partitioner());
  });
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

void EltwiseAddModAVX512(uint64_t* result, const uint64_t* operand1,
                         const uint64_t operand2, uint64_t n,
                         uint64_t modulus) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 63), "Require modulus < 2**63");
  HEXL_CHECK_BOUNDS(operand1, n, modulus,
                    "pre-add value in operand1 exceeds bound " << modulus);
  HEXL_CHECK(operand2 < modulus, "Require operand2 < modulus");

  uint64_t n_mod_8 = n % 8;
  if (n_mod_8 != 0) {
    EltwiseAddModNative(result, operand1, operand2, n_mod_8, modulus);
    operand1 += n_mod_8;
    result += n_mod_8;
    n -= n_mod_8;
  }

  __m512i v_modulus = _mm512_set1_epi64(static_cast<int64_t>(modulus));
  __m512i* vp_result = reinterpret_cast<__m512i*>(result);
  const __m512i* vp_operand1 = reinterpret_cast<const __m512i*>(operand1);
  const __m512i v_operand2 = _mm512_set1_epi64(static_cast<int64_t>(operand2));

  // std::cout << "n " << n << std::endl;
  omp_set_num_threads(2);
#pragma omp parallel firstprivate(vp_operand1, vp_result)
  {
    int id = omp_get_thread_num();
    int threads = omp_get_num_threads();
    vp_result += id * n / 8 / threads;
    vp_operand1 += id * n / 8 / threads;
    HEXL_LOOP_UNROLL_4
    for (size_t i = n / 8 / threads; i > 0; --i) {
      __m512i v_operand1 = _mm512_loadu_si512(vp_operand1);

      __m512i v_result =
          _mm512_hexl_small_add_mod_epi64(v_operand1, v_operand2, v_modulus);

      _mm512_storeu_si512(vp_result, v_result);

      ++vp_result;
      ++vp_operand1;
    }
  }
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

}  // namespace hexl
}  // namespace intel

#endif
