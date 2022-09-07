// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-add-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>
#include <unistd.h>
#include <sched.h>

#include <chrono>
#include <thread>

#include <iostream>

#include "eltwise/eltwise-add-mod-internal.hpp"
#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/thread-pool.hpp"
#include "util/avx512-util.hpp"

#ifdef HEXL_HAS_AVX512DQ

namespace intel {
namespace hexl {

const int thread_num = 32;

void EltwiseAddModAVX512_TP(uint64_t* result, const uint64_t* operand1,
                            const uint64_t* operand2, uint64_t n,
                            uint64_t modulus) {
  // std::cout << "ROCHA Add Mod" << std::endl;

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

  /*
  static int call = 0;

  ThreadPoolExecutor::SetNumberOfThreads(6);

  ThreadPoolExecutor::AddParallelTask([=]() {
    sleep(ThreadPoolExecutor::GetThreadId());
    std::cout << "ROCHA INSIDE " << call << " " << (*vp_operand1)[0]
              << std::endl;
    std::cout << "ID " << std::this_thread::get_id() << " ID("
              << ThreadPoolExecutor::GetThreadId() << ")" << std::endl;
    sleep(ThreadPoolExecutor::GetNumberOfThreads() -
          ThreadPoolExecutor::GetThreadId());
    // std::cout << "Total " << ThreadPoolExecutor::GetNumberOfThreads() <<
    // std::endl;
  });

  // std::cout << "ROCHA WAITING" << std::endl;
  ThreadPoolExecutor::SetBarrier();
  call++;
  */

  // std::cout << "ROCHA Jobs Launched" << std::endl;
  ThreadPoolExecutor::SetNumberOfThreads(eltwise_num_threads);
  // std::cout << "ROCHA call " << n << std::endl;
   /*
  ThreadPoolExecutor::AddParallelTask([vp_result,n, vp_operand1, vp_operand2, v_modulus](s_thread_info_t* thread_handler) {
    //int id = ThreadPoolExecutor::GetThreadId();
    int64_t id = thread_handler->thread_id;
    //int threads = ThreadPoolExecutor::GetNumberOfThreads();
    int64_t threads = thread_handler->total_threads; 
    __m512i* i_vp_result = vp_result + id * n / 8 / threads;
    // std::this_thread::sleep_for(std::chrono::nanoseconds(600));
    //std::cout << "ROCHA id on CPU " << sched_getcpu() << std::endl;
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
  });*/

  ThreadPoolExecutor::SetBarrier();

  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
  // std::cout << "ROCHA Finished Add Mod" << std::endl;
}

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
  //omp_set_num_threads(34);
#pragma omp parallel num_threads(eltwise_num_threads) firstprivate(vp_operand1, vp_operand2, vp_result)
  {
    int id = omp_get_thread_num();
    int threads = omp_get_num_threads();
    //std::cout << "ROCHA: on CPU " << sched_getcpu() << std::endl;
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
  //omp_set_num_threads(32);
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

static const size_t N = 23;

/*
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
  oneapi::tbb::task_arena arena(thread_num);
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
        }
        *//*
        ParallelEltwiseAddModAVX512(vp_result, vp_operand1, vp_operand2,
                                    v_modulus),
        tbb::static_partitioner());
  });
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}
*/

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
  //omp_set_num_threads(34);
#pragma omp parallel num_threads(eltwise_num_threads) firstprivate(vp_operand1, vp_result)
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
  //omp_set_num_threads(32);
  HEXL_CHECK_BOUNDS(result, n, modulus, "result exceeds bound " << modulus);
}

}  // namespace hexl
}  // namespace intel

#endif
