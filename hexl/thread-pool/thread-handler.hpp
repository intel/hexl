// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
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

using Task = std::function<void(size_t id, size_t threads)>;

// Controls thread
class ThreadHandler {
 public:
  // Control variables
  std::atomic<STATE> state{STATE::NONE};  // Keeps thread's state
  std::condition_variable waker;          // To wake thread up
  std::mutex wake_mutex;                  // For cond. variable
  std::thread thread;
  Task task;                                      // To be run by thread
  size_t thread_id;                               // Used for proper chunking
  inline static thread_local bool isChildThread;  // True if on child thread

  // Constructor
  ThreadHandler(const std::vector<ThreadHandler*>& handlers, size_t id)
      : thread_id(id) {
    thread = std::thread(&ThreadHandler::runner, this, std::cref(handlers));
  }

  // Thread Runner
  void runner(const std::vector<ThreadHandler*>& parent_container) {
    // This is a child thread
    ThreadHandler::isChildThread = true;
    // Handling loop
    while (true) {
      // Set thread ready
      state.store(STATE::DONE);

      // Timestamp: start active waiting
      auto spin_start = std::chrono::steady_clock::now();
      // Active waiting for KICK_OFF (or KILL)
      while (state.load() != STATE::KICK_OFF) {
        // Terminate thread?
        if (state.load() == STATE::KILL) {
          stop_thread = true;
          break;
        }
        // Go to sleep?
        if (elapsed_time(spin_start) > HEXL_THREAD_WAIT_TIME) {
          // Sleep waiting
          wait_for_wakeup();
          // break;
        }
      }

      if (stop_thread) break;  // Finish handling function

      // Specify the task its thread id and the total number of threads
      // at the time the task is run because thread pool size can change
      // while existing threads are running.
      state.store(STATE::RUNNING);
      task(thread_id, parent_container.size());
    }
  }

 private:
  bool stop_thread = false;  // To exit main thread loop

  // Waits for wake up signal on conditional variable
  void wait_for_wakeup() {
    // Protects against spurious wakeups
    auto predicate = [&rstate = state]() {
      return (rstate.load() != STATE::SLEEPING);
    };

    std::unique_lock<std::mutex> lock{wake_mutex};

    STATE current_state = STATE::DONE;
    STATE desired_state = STATE::SLEEPING;
    if (state.compare_exchange_strong(current_state, desired_state,
                                      std::memory_order_release,
                                      std::memory_order_relaxed)) {
      waker.wait(lock, predicate);
    }
  }

  // Returns elapsed time from given start time
  uint64_t elapsed_time(
      std::chrono::time_point<std::chrono::steady_clock> since) {
    // Timestamp: Current active waiting time
    auto spin_current = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(spin_current -
                                                                 since)
        .count();
  }
};

}  // namespace hexl
}  // namespace intel
