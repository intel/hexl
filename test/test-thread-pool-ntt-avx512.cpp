// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifdef HEXL_MULTI_THREADING

#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#include "test/test-ntt-util.hpp"
#include "test/test-thread-pool-util.hpp"
#include "test/test-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

class ParallelNTTCalls : public ParallelRecursion {};

#ifdef HEXL_HAS_AVX512DQ

TEST_P(ParallelNTTCalls, Stress) {
  uint64_t depth = GetParam();
  uint64_t nthreads = (1ULL << (depth + 1)) - 2;
  if (!has_avx512dq || nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }

  uint64_t m_N = 16384;
  uint64_t m_modulus = GeneratePrimes(1, 60, true, m_N)[0];
  NTT m_ntt = NTT(m_N, m_modulus);
  auto NTTCall = [=]() {
    for (size_t trial = 0; trial < m_num_trials; ++trial) {
      AlignedVector64<uint64_t> original =
          GenerateInsecureUniformIntRandomValues(m_N, 0, m_modulus);
      AlignedVector64<uint64_t> input_avx = original;

      ForwardTransformToBitReverseAVX512<64>(
          input_avx.data(), input_avx.data(), m_N, m_ntt.GetModulus(),
          m_ntt.GetAVX512RootOfUnityPowers().data(),
          m_ntt.GetAVX512Precon64RootOfUnityPowers().data(), 2, 1);

      InverseTransformFromBitReverseAVX512<64>(
          input_avx.data(), input_avx.data(), m_N, m_ntt.GetModulus(),
          m_ntt.GetInvRootOfUnityPowers().data(),
          m_ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

      ASSERT_EQ(original, input_avx);
    }
  };

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  HEXL_NTT_PARALLEL_DEPTH = depth;

  std::thread thread_object1(NTTCall);
  std::thread thread_object2(NTTCall);

  thread_object1.join();
  thread_object2.join();

  ThreadPoolExecutor::SetNumberOfThreads(0);
  HEXL_NTT_PARALLEL_DEPTH = 1;
}

INSTANTIATE_TEST_SUITE_P(ThreadPool, ParallelNTTCalls,
                         ::testing::Values(0, 1, 2, 3, 4, 5));

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_MULTI_THREADING
