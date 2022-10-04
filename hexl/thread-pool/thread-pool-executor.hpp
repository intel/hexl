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
  static void SetNumberOfThreads(uint n_threads) {
    pool->SetupThreads(n_threads);
  }

  static void AddParallelJobs(
      std::function<void(size_t id, size_t threads)> job) {
    pool->AddParallelJobs(job);
  }

  static void AddTask(std::function<void(size_t id, size_t threads)> task) {
    pool->AddTask(task);
  }

  static size_t GetNumberOfThreads() { return pool->GetNumThreads(); }

  static void SetBarrier() { pool->SetBarrier(); }

  static void StopThreads() { pool->StopThreads(); }

  static std::vector<const thread_info_t*> GetThreadHandlers() {
    return pool->GetThreadHandlers();
  }

  ~ThreadPoolExecutor() { delete pool; }

#else

 public:
  static void SetNumberOfThreads(int n_threads) { HEXL_UNUSED(n_threads); }

  static void AddParallelJobs(std::function<void(int id, int threads)> job) {
    job(0, 1);
  }

  static void AddTask(std::function<void(int id, int threads)> task) {
    task(0, 1);
  }

  static size_t GetNumberOfThreads() { return 1; }

  static void SetBarrier() {}

  static void StopThreads() {}

#endif
};

}  // namespace hexl
}  // namespace intel
