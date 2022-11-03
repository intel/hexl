// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <list>
#include <mutex>
#include <thread>

#include "hexl/util/check.hpp"
#include "thread-pool/thread-pool-executor.hpp"

namespace intel {
namespace hexl {

// Common task
constexpr uint64_t work_delay = 2;
constexpr uint64_t N_size = 100;
std::mutex tasks_mutex;
std::list<std::thread::id> task_ids;
std::atomic_int sync;
std::atomic_int iterations;

void dummy_task(int id, int threads) {
  HEXL_UNUSED(id);
  HEXL_UNUSED(threads);
}

void working_task(size_t id, size_t threads) {
  HEXL_UNUSED(id);
  HEXL_UNUSED(threads);
  std::this_thread::sleep_for(std::chrono::milliseconds(work_delay));
}

void id_task(int id, int threads) {
  HEXL_UNUSED(id);
  HEXL_UNUSED(threads);
  std::lock_guard<std::mutex> lock(tasks_mutex);
  task_ids.push_back(std::this_thread::get_id());
}

void add_iterations(int id, int threads) {
  HEXL_UNUSED(id);
  iterations.fetch_add(N_size / threads);
}

void recursive_calls(uint64_t depth, uint64_t level, uint64_t half) {
  if (level < depth) {
    ThreadPoolExecutor::AddRecursiveCalls(
        level, half,
        [&](int id, int threads) {
          HEXL_UNUSED(id);
          HEXL_UNUSED(threads);
          recursive_calls(depth, level + 1, 2 * half);
        },
        [&](int id, int threads) {
          HEXL_UNUSED(id);
          HEXL_UNUSED(threads);
          recursive_calls(depth, level + 1, 2 * half + 1);
        });
  }
  {
    std::lock_guard<std::mutex> lock(tasks_mutex);
    task_ids.push_back(std::this_thread::get_id());
    std::this_thread::sleep_for(std::chrono::microseconds(1));
  }
}

}  // namespace hexl
}  // namespace intel
