// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "thread-pool/thread-pool-util.hpp"

namespace intel {
namespace hexl {

using std::chrono::duration_cast;
typedef std::function<void(size_t id, size_t threads)> tp_task_t;

class ThreadPool {
 private:
  // Properties
  size_t total_threads = 0;                     // Total number of threads
  size_t next_thread = 0;                       // Points to next free thread
  std::vector<thread_info_t*> thread_handlers;  // Thread's info
  std::mutex pool_mutex;                        // Control pool's edition
  bool setup_done = false;                      // Is thread pool initialized

  // Methods
  // StartThreads: Spawn a given number of threads
  void StartThreads(size_t new_threads) {
    size_t current_threads = total_threads;

    // Add N new threads
    for (size_t i = 0; i < new_threads; ++i) {
      // Create handler and add it to vector of threads
      thread_info_t* thread_handler = new thread_info_t();
      thread_handlers.emplace_back(thread_handler);

      // Run thread with a handling function
      size_t* n_threads = &total_threads;  // To put within scope
      thread_handler->thread =
          std::thread([thread_handler, current_threads, i, n_threads] {
            bool stop = false;

            while (true) {
              // Thread ready
              thread_handler->state.store(STATE::DONE);

              // Start waiting timestamp
              auto spin_start = std::chrono::steady_clock::now();

              // Thread waiting
              while (true) {
                // Start or stop thread
                if (thread_handler->state.load() == STATE::KICK_OFF) break;
                if (thread_handler->state.load() == STATE::KILL) {
                  stop = true;
                  break;
                }

                // Check waiting time
                auto spin_current = std::chrono::steady_clock::now();
                uint64_t duration = duration_cast<std::chrono::milliseconds>(
                                        spin_current - spin_start)
                                        .count();
                if (duration > HEXL_THREAD_WAIT_TIME) {
                  // Got to sleep mode
                  thread_handler->state.store(STATE::SLEEPING);

                  // Wait for start or stop
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

              if (stop) break;  // Finish handling function

              // Thread is running task
              thread_handler->state.store(STATE::RUNNING);
              thread_handler->task(current_threads + i, *n_threads);
            }
          });
    }

    // New threads added
    total_threads += new_threads;
  }

  // WaitThread: Wait for one thread to be ready
  void WaitThread(thread_info_t* thread_handler) {
    while (1) {
      if (thread_handler->state.load() == STATE::DONE ||
          thread_handler->state.load() == STATE::SLEEPING) {
        break;
      }
    }
  }

  // SetBarrier_Unlocked: Sets a barrier to sync all threads. Without mutex
  void SetBarrier_Unlocked() {
    // Waits until all threads are DONE or SLEEPING
    for (size_t i = 0; i < total_threads; i++) {
      WaitThread(thread_handlers.at(i));
    }

    // Next thread to be used
    next_thread = 0;
  }

  // SetupThreads_Unlocked: Spawns new threads if necessary. Without mutex
  void SetupThreads_Unlocked(size_t n_threads) {
    HEXL_VLOG(3, "Thread Pool Info:");
    HEXL_VLOG(3, "HEXL_NUM_THREADS                = " << HEXL_NUM_THREADS);
    HEXL_VLOG(3,
              "HEXL_NTT_PARALLEL_DEPTH         = " << HEXL_NTT_PARALLEL_DEPTH);

    // Add new threads if necessary
    if (total_threads < n_threads) {
      // Can't exceed HW threads
      if (n_threads > std::thread::hardware_concurrency()) {
        n_threads = std::thread::hardware_concurrency();
        HEXL_VLOG(
            3, "Exceeded platform's available number of threads. Setting to: "
                   << std::thread::hardware_concurrency() << ".");
      }

      size_t new_threads = n_threads - total_threads;
      StartThreads(new_threads);  // Start extra threads

      // Wait for threads to be ready
      SetBarrier_Unlocked();
    }

    // Thread pool up
    setup_done = true;

    HEXL_VLOG(2,
              "Setting up thread pool with " << total_threads << " threads.");
  }

 public:
  // Methods

  ThreadPool() {}

  ~ThreadPool() { StopThreads(); }

  // GetNumThreads: Returns total number of threads
  size_t GetNumThreads() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    return total_threads;
  }

  // AddParallelJobs: Runs the same function on a total number of threads
  void AddParallelJobs(tp_task_t job) {
    HEXL_CHECK(job, "Require non empty job");

    // Try using thread pool
    if (pool_mutex.try_lock()) {
      if (!setup_done) {
        SetupThreads_Unlocked(HEXL_NUM_THREADS);  // Setup if thread pool down
      }

      if (next_thread == 0) {  // Only if all threads are available
        for (size_t i = 0; i < total_threads; i++) {
          thread_info_t* thread_handler = thread_handlers.at(i);
          if (thread_handler->state.load() == STATE::DONE) {
            thread_handler->task = job;
            thread_handler->state.store(STATE::KICK_OFF);
          } else if (thread_handler->state.load() == STATE::SLEEPING) {
            thread_handler->task = job;
            thread_handler->state.store(STATE::KICK_OFF);
            thread_handler->waker.notify_one();
          } else {  // In case thread is not on expected state
            job(i, total_threads);
          }
        }
        next_thread = total_threads;

        SetBarrier_Unlocked();  // Wait 'til all jobs are done
        pool_mutex.unlock();
      } else {  // Run on single thread
        pool_mutex.unlock();
        job(0, 1);
      }
    } else {
      job(0, 1);
    }
  }

  // AddTask: Runs a task on next thread available
  void AddTask(tp_task_t task) {
    HEXL_CHECK(task, "Require non empty task");

    // Try using thread pool
    if (pool_mutex.try_lock()) {
      if (!setup_done) {
        SetupThreads_Unlocked(HEXL_NUM_THREADS);  // Setup if thread pool down
      }

      size_t next = next_thread++;  // Get next thread to be used
      if (next < total_threads) {
        thread_info_t* thread_handler = thread_handlers.at(next);
        if (thread_handler->state.load() == STATE::DONE) {
          thread_handler->task = task;
          thread_handler->state.store(STATE::KICK_OFF);
          pool_mutex.unlock();
        } else if (thread_handler->state.load() == STATE::SLEEPING) {
          thread_handler->task = task;
          thread_handler->state.store(STATE::KICK_OFF);
          thread_handler->waker.notify_one();
          pool_mutex.unlock();
        } else {  // In case thread is not on expected state
          pool_mutex.unlock();
          task(next - 1, total_threads);
        }
      } else {
        pool_mutex.unlock();
        task(0, 1);
      }
    } else {
      task(0, 1);
    }
  }

  // SetupThreads: Spawns new threads if necessary
  void SetupThreads(size_t n_threads) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    SetupThreads_Unlocked(n_threads);
  }

  // WaitThreads: Sets a barrier to sync all threads with mutex
  void SetBarrier() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    SetBarrier_Unlocked();
  }

  // Stop threads
  void StopThreads() {
    std::lock_guard<std::mutex> lock(pool_mutex);

    SetBarrier_Unlocked();  // Waits for threads to be ready

    for (size_t i = 0; i < total_threads; i++) {
      thread_info_t* thread_handler = thread_handlers.at(i);
      if (thread_handler->state.load() == STATE::SLEEPING) {
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

  // Return threads' handlers
  std::vector<const thread_info_t*> GetThreadHandlers() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    std::vector<const thread_info_t*> handlers;
    handlers.assign(thread_handlers.begin(), thread_handlers.end());
    return handlers;
  }
};

}  // namespace hexl
}  // namespace intel
