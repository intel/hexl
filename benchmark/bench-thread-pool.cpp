// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include <chrono>
#include <ctime>

#include "thread-pool/thread-pool-executor.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_MULTI_THREADING

// state[0] is the threads
static void BM_ThreadPool_SetupJoin(benchmark::State& state) {  //  NOLINT
  size_t threads = state.range(0);
  if (threads > std::thread::hardware_concurrency()) {
    state.SkipWithError("No threads available");
  }

  for (auto _ : state) {
    ThreadPoolExecutor::SetNumberOfThreads(threads);
    ThreadPoolExecutor::SetNumberOfThreads(0);
  }
}

BENCHMARK(BM_ThreadPool_SetupJoin)
    ->Unit(benchmark::kMicrosecond)
    ->Args({2})
    ->Args({8})
    ->Args({16});

// state[0] is the threads
static void BM_ThreadPool_WakeUp_Plus30ms(benchmark::State& state) {  //  NOLINT
  size_t threads = state.range(0);
  if (threads > std::thread::hardware_concurrency()) {
    state.SkipWithError("No threads available");
  }

  ThreadPoolExecutor::SetNumberOfThreads(threads);

  for (auto _ : state) {
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    ThreadPoolExecutor::AddParallelJobs(0, [](size_t start, size_t end) {
      HEXL_UNUSED(start);
      HEXL_UNUSED(end);
    });
  }
  ThreadPoolExecutor::SetNumberOfThreads(0);
}

BENCHMARK(BM_ThreadPool_WakeUp_Plus30ms)
    ->Unit(benchmark::kMicrosecond)
    ->Args({2})
    ->Args({8})
    ->Args({16});

#endif

}  // namespace hexl
}  // namespace intel
