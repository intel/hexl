// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>

#include "hexl/thread-pool/thread-pool-vars.hpp"
#include "thread-pool/thread-pool.hpp"

namespace intel {
namespace hexl {

class ThreadPoolExecutor {
#ifdef HEXL_MULTI_THREADING

 private:
  inline static ThreadPool* pool = new ThreadPool(HEXL_NUM_THREADS);

 public:
  static void SetNumberOfThreads(int n_threads) {
    pool->SetupThreads(n_threads);
  }

  static void AddParallelJobs(std::function<void(int id, int threads)> job) {
    pool->AddParallelJobs(job);
  }

  static void AddTask(std::function<void(int id, int threads)> task) {
    pool->AddTask(task);
  }

  static size_t GetNumberOfThreads() { return pool->GetNumThreads(); }

  static void SetBarrier() { pool->WaitThreads(); }
#else

 public:
  static void SetNumberOfThreads(int n_threads) { HEXL_UNUSED(n_threads); }

  static void AddParallelJobs(std::function<void(int id, int threads)> job) {
    std::cout << "SINGLE" << std::endl;
    job(0, 1);
  }

  static void AddTask(std::function<void(int id, int threads)> task) {
    task(0, 1);
  }

  static size_t GetNumberOfThreads() { return 1; }

  static void SetBarrier() {}
#endif
};

}  // namespace hexl
}  // namespace intel
