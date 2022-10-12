// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace intel {
namespace hexl {

// Enum for thread states
enum class STATE : int {
  NONE = 0,      // Undefined state
  DONE = 1,      // Task is completed and thread is on spin-up
  KICK_OFF = 2,  // There is a new task to execute, break spin-up
  RUNNING = 3,   // Executing task
  SLEEPING = 4,  // Thread is sleeping, waiting for wakeup
  KILL = 5       // To join thread
};

// Control variables per thread
struct ThreadInfo {
  std::atomic<STATE> state{STATE::NONE};
  std::condition_variable waker;
  std::mutex wake_mutex;
  std::thread thread;
  std::function<void(size_t id, size_t threads)> task;
};

}  // namespace hexl
}  // namespace intel
