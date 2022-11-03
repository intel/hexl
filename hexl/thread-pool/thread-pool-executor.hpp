// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/thread-pool/thread-pool-vars.hpp"
#include "thread-pool/thread-pool.hpp"

namespace intel {
namespace hexl {

class ThreadPoolExecutor {
#ifdef HEXL_MULTI_THREADING

 private:
  inline static ThreadPool pool{};

 public:
  // SetNumberOfThreads: Setup/kill thread pool by specifying number of threads
  static void SetNumberOfThreads(uint64_t n_threads) {
    pool.SetupThreads(n_threads);
  }

  // AddParallelJobs: For parallel loops
  static void AddParallelJobs(Task job) { pool.AddParallelJobs(job); }

  // AddRecursiveCalls: For parallel recursion
  static void AddRecursiveCalls(uint64_t depth, uint64_t half, Task task_a,
                                Task task_b) {
    pool.AddRecursiveCalls(depth, half, task_a, task_b);
  }

  // Return total number of threads
  static size_t GetNumberOfThreads() { return pool.GetNumThreads(); }

  // Return vector of constant handlers
  static std::vector<const ThreadHandler*> GetThreadHandlers() {
    return pool.GetThreadHandlers();
  }

#else

 public:
  static void SetNumberOfThreads(int n_threads) { HEXL_UNUSED(n_threads); }

  static void AddParallelJobs(Task job) { job(0, 1); }

  static void AddRecursiveCalls(uint64_t depth, uint64_t half, Task task_a,
                                Task task_b) {
    HEXL_UNUSED(depth);
    HEXL_UNUSED(half);
    task_a(0, 1);
    task_b(0, 1);
  }

  static size_t GetNumberOfThreads() { return 1; }

#endif
};

}  // namespace hexl
}  // namespace intel
