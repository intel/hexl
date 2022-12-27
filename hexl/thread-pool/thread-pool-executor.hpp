// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "thread-pool/thread-pool.hpp"

namespace intel {
namespace hexl {

class ThreadPoolExecutor {
#ifdef HEXL_MULTI_THREADING

 private:
  inline static ThreadPool pool{};

 public:
  // SetNumberOfThreads: Sets up/down thread pool by specified number of threads
  static void SetNumberOfThreads(size_t n_threads) {
    pool.SetupThreadPool(n_threads);
  }

  // SetNumberOfThreads: Sets up/down threads by specified number of threads.
  // Sets parallel recursive depth
  static void SetNumberOfThreadsAndDepth(size_t n_threads,
                                         size_t parallel_depth) {
    pool.SetupThreadPool(n_threads, parallel_depth);
  }

  // AddParallelJobs: For parallel loops
  static void AddParallelJobs(size_t N, Task job) {
    pool.AddParallelJobs(N, job);
  }

  // AddRecursiveCalls: For parallel recursion
  static void AddRecursiveCalls(size_t depth, size_t half, Task task_a,
                                Task task_b) {
    pool.AddRecursiveCalls(depth, half, task_a, task_b);
  }

  // Return total number of threads
  static size_t GetNumberOfThreads() { return pool.GetNumThreads(); }

  // Return parallel recursive depth
  static size_t GetParallelDepth() { return pool.GetParallelDepth(); }

  // Return vector of constant handlers
  static std::vector<const ThreadHandler*> GetThreadHandlers() {
    return pool.GetThreadHandlers();
  }

#else

 public:
  static void SetNumberOfThreads(size_t n_threads) { HEXL_UNUSED(n_threads); }

  static void SetNumberOfThreadsAndDepth(size_t n_threads,
                                         size_t parallel_depth) {
    HEXL_UNUSED(n_threads);
    HEXL_UNUSED(parallel_depth);
  }

  static void AddParallelJobs(size_t N, Task job) { job(0, N); }

  static void AddRecursiveCalls(size_t depth, size_t half, Task task_a,
                                Task task_b) {
    HEXL_UNUSED(depth);
    HEXL_UNUSED(half);
    task_a(0, 0);
    task_b(0, 0);
  }

  static size_t GetNumberOfThreads() { return 1; }

  static size_t GetParallelDepth() { return 0; }

#endif
};

}  // namespace hexl
}  // namespace intel
