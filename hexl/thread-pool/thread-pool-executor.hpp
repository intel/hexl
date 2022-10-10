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
  inline static ThreadPool* pool = new ThreadPool();

 public:
  // SetNumberOfThreads: Setup/kill thread pool by specifying number of threads
  static void SetNumberOfThreads(uint64_t n_threads) {
    pool->SetupThreads(n_threads);
  }

  // AddParallelJobs: For parallel loops
  static void AddParallelJobs(tp_task_t job) { pool->AddParallelJobs(job); }

  // AddRecursiveCalls: For parallel recursion
  static void AddRecursiveCalls(tp_task_t task_a, tp_task_t task_b) {
    pool->AddRecursiveCalls(task_a, task_b);
  }

  // Return total number of threads
  static size_t GetNumberOfThreads() { return pool->GetNumThreads(); }

  // Return vector of constant handlers
  static std::vector<const thread_info_t*> GetThreadHandlers() {
    return pool->GetThreadHandlers();
  }

  // Destructor
  ~ThreadPoolExecutor() { delete pool; }

#else

 public:
  static void SetNumberOfThreads(int n_threads) { HEXL_UNUSED(n_threads); }

  static void AddParallelJobs(tp_task_t job) { job(0, 1); }

  static void AddRecursiveCalls(tp_task_t task_a, tp_task_t task_b) {
    task_a(0, 1);
    task_b(0, 1);
  }

  static size_t GetNumberOfThreads() { return 1; }

#endif
};

}  // namespace hexl
}  // namespace intel
