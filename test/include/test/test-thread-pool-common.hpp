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

#ifdef HEXL_MULTI_THREADING

namespace intel {
namespace hexl {

// Parameters for testing
static constexpr uint64_t work_delay = 2;
static std::mutex tasks_mutex;
static std::list<std::thread::id> task_ids;
static std::atomic_int sync;
static std::atomic_int iterations;

// Common tasks
static void dummy_task(int start, int end) {
  HEXL_UNUSED(start);
  HEXL_UNUSED(end);
}

static void working_task(size_t start, size_t end) {
  HEXL_UNUSED(start);
  HEXL_UNUSED(end);
  std::this_thread::sleep_for(std::chrono::milliseconds(work_delay));
}

static void id_task(int start, int end) {
  HEXL_UNUSED(start);
  HEXL_UNUSED(end);
  {
    std::lock_guard<std::mutex> lock(tasks_mutex);
    task_ids.push_back(std::this_thread::get_id());
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(work_delay >> 1));
}

static void add_iterations(int start, int end) {
  iterations.fetch_add(end - start);
  std::this_thread::sleep_for(std::chrono::milliseconds(work_delay >> 1));
}

static void recursive_calls(uint64_t delay, uint64_t depth, uint64_t level,
                            uint64_t half) {
  if (level < depth) {
    ThreadPoolExecutor::AddRecursiveCalls(
        level, half,
        [&](int start, int end) {
          HEXL_UNUSED(start);
          HEXL_UNUSED(end);
          recursive_calls(delay, depth, level + 1, 2 * half);
        },
        [&](int start, int end) {
          HEXL_UNUSED(start);
          HEXL_UNUSED(end);
          recursive_calls(delay, depth, level + 1, 2 * half + 1);
        });
  } else {
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  }

  {
    std::lock_guard<std::mutex> lock(tasks_mutex);
    task_ids.push_back(std::this_thread::get_id());
  }
}

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_MULTI_THREADING
