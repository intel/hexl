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
enum STATE {
  NONE,      // Undefined state
  DONE,      // Task is completed and thread is on spin-up
  KICK_OFF,  // There is a new task to execute, break spin-up
  RUNNING,   // Executing task
  SLEEPING,  // Thread is sleeping, waiting for wakeup
  KILL       // To join thread
};

// Control variables per thread
struct thread_info_t {
  std::atomic_int state{STATE::NONE};
  std::condition_variable waker;
  std::mutex wake_mutex;
  std::thread thread;
  std::function<void(size_t id, size_t threads)> task;
};

}  // namespace hexl
}  // namespace intel
