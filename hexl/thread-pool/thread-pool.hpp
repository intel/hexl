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
#include "hexl/util/check.hpp"
#include "thread-pool/thread-pool-util.hpp"

namespace intel {
namespace hexl {

using std::chrono::duration_cast;

class ThreadPool {
 private:
  // Properties
  uint total_threads = 0;                       // Total number of threads
  std::atomic_int next_thread{0};               // Points to next free thread
  std::vector<thread_info_t*> thread_handlers;  // Thread's info
  bool setup_done = false;

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
            bool stop = false;
            while (true) {
              // std::cout << "ROCHA: Thread waiting. Thread ID " <<
              // current_threads + i << std::endl;
              thread_handler->state.store(STATE::DONE);
              auto spin_start = std::chrono::steady_clock::now();
              while (true) {  // Thread spin-up
                // Start or stop thread
                if (thread_handler->state.load() == STATE::KICK_OFF) break;
                if (thread_handler->state.load() == STATE::KILL) {
                  stop = true;
                  break;
                }
                // Enter sleep mode if idle
                auto spin_current = std::chrono::steady_clock::now();
                uint64_t duration = duration_cast<std::chrono::milliseconds>(
                                        spin_current - spin_start)
                                        .count();
                // std::cout << "Duration " << duration << std::endl;
                if (duration > HEXL_THREAD_WAIT_TIME) {
                  // std::cout << "ROCHA sleep  " << current_threads + i << " d
                  // " << HEXL_THREAD_WAIT_TIME << std::endl;
                  thread_handler->state.store(STATE::SLEEPING);
                  std::unique_lock<std::mutex> lock{thread_handler->wake_mutex};
                  thread_handler->waker.wait(lock, [&stop, thread_handler] {
                    if (thread_handler->state.load() == STATE::KICK_OFF) {
                      return true;
                    }
                    if (thread_handler->state.load() == STATE::KILL) {
                      stop = true;
                      return true;
                    }
                    return false;
                  });
                  break;
                }
              }

              if (stop) break;

              // std::cout << "ROCHA: Thread working. Thread ID " <<
              // current_threads + i << std::endl;
              thread_handler->state.store(STATE::RUNNING);
              thread_handler->task(current_threads + i, *threads);
            }
          });
    }
    total_threads += new_threads;
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

  // AddParallelJobs: Runs the same function on a total number of threads
  void AddParallelJobs(std::function<void(int id, int threads)> job) {
    HEXL_CHECK(job, "Require non empty job");

    if (!setup_done) {
      SetupThreads(HEXL_NUM_THREADS);
      setup_done = true;
    }

    // std::cout << "ROCHA Added Job" << std::endl;
    if (next_thread.load() == 0) {  // If all threads available
      for (uint i = 0; i < total_threads; i++) {
        thread_info_t* thread_handler = thread_handlers.at(i);
        thread_handler->task = job;
        if (thread_handler->state.load() == STATE::DONE) {
          thread_handler->state.store(STATE::KICK_OFF);
        } else if (thread_handler->state.load() == STATE::SLEEPING) {
          // std::cout << "Waking up " << i << std::endl;
          thread_handler->state.store(STATE::KICK_OFF);
          thread_handler->waker.notify_one();
        } else {
          job(i, total_threads);
        }
        // std::cout << "ROCHA Added Job. ID: " << i << std::endl;
      }
      next_thread.store(total_threads);

      WaitThreads();  // Wait till all job is done

    } else {  // Run on single thread
      job(0, 1);
    }
  }

  // AddTask: Runs a task on next thread available
  void AddTask(std::function<void(int id, int threads)> task) {
    HEXL_CHECK(task, "Require non empty task");

    if (!setup_done) {
      SetupThreads(HEXL_NUM_THREADS);
      setup_done = true;
    }

    // std::cout << "ROCHA AddTask" << std::endl;
    uint next = next_thread.fetch_add(1);
    if (next < total_threads) {
      // std::cout << "ROCHA Task Adding " << next << std::endl;
      thread_info_t* thread_handler = thread_handlers.at(next);
      thread_handler->task = task;
      if (thread_handler->state.load() == STATE::DONE) {
        thread_handler->state.store(STATE::KICK_OFF);
      } else if (thread_handler->state.load() == STATE::SLEEPING) {
        // std::cout << "Waking up " << i << std::endl;
        thread_handler->state.store(STATE::KICK_OFF);
        thread_handler->waker.notify_one();
      } else {
        task(next - 1, total_threads);
      }
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
        // HEXL_VLOG(
        //     3, "Exceeded platform's available number of threads. Setting to:
        //     "
        //            << std::thread::hardware_concurrency() << ".");
      }

      // std::cout << "ROCHA Setup " << n_threads << " threads" << std::endl;
      int new_threads = n_threads - total_threads;
      StartThreads(new_threads);

      // Wait threads are ready
      WaitThreads();
    }

    setup_done = true;
  }

  // WaitThreads: Sets a barrier to sync all threads
  void WaitThreads() {
    // std::cout << "ROCHA Barrier on " << num_threads << " threads" <<
    // std::endl;
    for (uint i = 0; i < total_threads; i++) {
      thread_info_t* thread_handler = thread_handlers.at(i);
      while (1) {
        if (thread_handler->state.load() == STATE::DONE ||
            thread_handler->state.load() == STATE::SLEEPING) {
          // std::cout << "ROCHA Thread "<< i << " Done" << std::endl;
          break;
        }
      }
    }
    next_thread.store(0);
    // std::cout << "ROCHA Barrier Done" << std::endl;
  }

  // Stop threads
  void StopThreads() {
    WaitThreads();

    for (uint i = 0; i < total_threads; i++) {
      thread_info_t* thread_handler = thread_handlers.at(i);
      if (thread_handler->state.load() == STATE::SLEEPING) {
        // std::cout << "Waking up " << i << std::endl;
        thread_handler->state.store(STATE::KILL);
        thread_handler->waker.notify_one();
      } else {
        thread_handler->state.store(STATE::KILL);
      }
      thread_handler->thread.join();
      delete thread_handler;
    }
    thread_handlers.clear();
    total_threads = 0;
    setup_done = false;
  }

  std::vector<const thread_info_t*> GetThreadHandlers() {
    std::vector<const thread_info_t*> handlers;
    handlers.assign(thread_handlers.begin(), thread_handlers.end());
    return handlers;
  }
};

}  // namespace hexl
}  // namespace intel
