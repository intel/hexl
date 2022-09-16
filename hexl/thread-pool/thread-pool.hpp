// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include <condition_variable>
// #include <cstddef>

#include <iostream>
// #include <map>
// #include <queue>

#include <vector>

#include "hexl/logging/logging.hpp"
#include "thread-pool/thread-pool-util.hpp"

namespace intel {
namespace hexl {

class ThreadPool {
 private:
  // Properties
  uint total_threads = 0;                       // Total number of threads
  std::atomic_int next_thread{0};               // Points to next free thread
  std::vector<thread_info_t*> thread_handlers;  // Thread's info

  // Methods
  // StartThreads: Spawn a given number of threads
  void StartThreads(int new_threads) {
    // std::cout << "ROCHA Start" << std::endl;
    int current_threads = total_threads;
    for (int i = 0; i < new_threads; ++i) {
      thread_info_t* thread_handler = new thread_info_t();
      thread_handlers.emplace_back(thread_handler);
      uint* threads = &total_threads;
      thread_handler->thread =
          std::thread([thread_handler, current_threads, i, threads] {
            while (true) {
              // std::cout << "ROCHA: Thread waiting. Thread ID " <<
              // current_threads + i << std::endl;
              thread_handler->state.store(STATE::DONE);
              while (1) {  // Thread spin-up
                if (thread_handler->state.load() == STATE::KICK_OFF) break;
              }
              // std::cout << "ROCHA: Thread working. Thread ID " <<
              // current_threads + i << std::endl;
              thread_handler->state.store(STATE::RUNNING);
              thread_handler->task(current_threads + i, *threads);
            }
          });
    }
    total_threads += new_threads;
    // WaitThreads();
  }

  // Stop threads
  void StopThreads() {
    std::cout << "ROCHA Stop" << std::endl;
    {
      // std::lock_guard<std::mutex> lock{event_lock};
      // stop_threads = true;
    }

    // wake_condition.notify_all();

    // for (auto& thread : threads) {
    //  thread.join();
    //}
  }

 public:
  // Methods

  explicit ThreadPool(int n_threads) {
    // std::cout << "ROCHA Thread pool constructed" << std::endl;
    std::cout << "ROCHA struct size:" << sizeof(thread_info_t) << " threads "
              << n_threads << std::endl;
    StartThreads(n_threads);
  }

  ~ThreadPool() {
    // std::cout << "ROCHA Thread pool destructor" << std::endl;
    StopThreads();
  }

  // GetNumThreads: Returns total number of threads
  size_t GetNumThreads() { return total_threads; }

  // AddParallelJobs: Runs the same function on total number of threads
  void AddParallelJobs(std::function<void(int id, int threads)> job) {
    // std::cout << "ROCHA Added Job" << std::endl;
    if (next_thread.load() == 0) {  // If all threads available
      for (uint i = 0; i < total_threads; i++) {
        thread_info_t* thread_handler = thread_handlers.at(i);
        {
          thread_handler->task = job;
          thread_handler->state.store(STATE::KICK_OFF);
          // std::cout << "ROCHA Added Job. ID: " << i << std::endl;
        }
      }
      next_thread.store(total_threads);
    } else {  // Run on single thread
      job(0, 1);
    }
  }

  // AddTask: Runs a task on next thread available
  void AddTask(std::function<void(int id, int threads)> task) {
    // std::cout << "ROCHA Added Job" << std::endl;
    uint next = next_thread.fetch_add(1);
    if (next < total_threads) {
      // std::cout << "ROCHA Task Adding " << std::endl;
      thread_info_t* thread_handler = thread_handlers.at(next);
      thread_handler->task = task;
      thread_handler->state.store(STATE::KICK_OFF);
      // std::cout << "ROCHA Task Added " << available << std::endl;
    } else {
      // std::cout << "ROCHA No available " << std::endl;
      task(0, 1);
    }
  }

  // SetupThreads: Spawns new threads if necessary
  void SetupThreads(uint n_threads) {
    // std::cout << "ROCHA Setup" << std::endl;
    if (total_threads < n_threads) {
      if (n_threads > std::thread::hardware_concurrency()) {
        n_threads = std::thread::hardware_concurrency();
        HEXL_VLOG(
            3, "Exceeded platform's available number of threads. Setting to: "
                   << std::thread::hardware_concurrency() << ".");
      }

      // std::cout << "ROCHA Setup " << n_threads << " threads" << std::endl;
      int new_threads = n_threads - total_threads;
      StartThreads(new_threads);
    }
  }

  // WaitThreads: Sets a barrier to sync all threads
  void WaitThreads() {
    // std::cout << "ROCHA Barrier on " << num_threads << " threads" <<
    // std::endl;
    for (uint i = 0; i < total_threads; i++) {
      thread_info_t* thread_handler = thread_handlers.at(i);
      while (1) {
        if (thread_handler->state.load() == STATE::DONE) {
          // std::cout << "ROCHA Thread "<< i << " Done" << std::endl;
          break;
        }
      }
    }
    next_thread.store(0);
    // std::cout << "ROCHA Barrier Done" << std::endl;
  }
};

}  // namespace hexl
}  // namespace intel
