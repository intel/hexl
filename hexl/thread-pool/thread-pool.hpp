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
using tp_task_t = std::function<void(size_t id, size_t threads)>;

class ThreadPool {
 private:
  // Properties
  size_t total_threads = 0;                  // Total number of threads
  std::atomic_uint64_t next_thread{0};       // Points to next free thread
  std::vector<ThreadInfo*> thread_handlers;  // Thread's info
  std::mutex pool_mutex;                     // Control pool's edition
  bool setup_done = false;                   // Is thread pool initialized
  inline static thread_local bool child;     // True if on child thread

  // Methods

  // StartThreads: Spawn a given number of threads
  void StartThreads(size_t new_threads) {
    size_t current_threads = total_threads;

    // Add N new threads
    for (size_t i = 0; i < new_threads; ++i) {
      // Create handler and add it to vector of threads
      ThreadInfo* thread_handler = new ThreadInfo();
      thread_handlers.emplace_back(thread_handler);

      // Run thread with a handling function
      // size_t* n_threads = &total_threads;  // To put within scope

      thread_handler->thread = std::thread(
          [thread_handler, current_threads, i, &n_threads = total_threads] {
            bool stop = false;
            ThreadPool::child = true;

            while (true) {
              // Thread ready
              thread_handler->state.store(STATE::DONE);

              // Start waiting timestamp
              auto spin_start = std::chrono::steady_clock::now();

              // Thread waiting
              while (thread_handler->state.load() != STATE::KICK_OFF) {
                // Stop thread
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
              thread_handler->task(current_threads + i, n_threads);
            }
          });
    }

    // New threads added
    total_threads += new_threads;
  }

  // WaitThread: Wait for one thread to be ready
  void WaitThread(ThreadInfo* thread_handler) {
    while (thread_handler->state.load() != STATE::DONE &&
           thread_handler->state.load() != STATE::SLEEPING) {
    }
  }

  // SetBarrier_Unlocked: Sets a barrier to sync all threads. Without mutex
  void SetBarrier_Unlocked() {
    // Waits until all threads are DONE or SLEEPING
    for (size_t i = 0; i < total_threads; i++) {
      WaitThread(thread_handlers.at(i));
    }

    // Next thread to be used
    next_thread.store(0);
  }

  // SetupThreads_Unlocked: Spawns new threads if necessary. Without mutex
  void SetupThreads_Unlocked(size_t n_threads) {
    HEXL_VLOG(3, "Thread Pool Info:");
    HEXL_VLOG(3, "HEXL_NUM_THREADS                = " << HEXL_NUM_THREADS);
    HEXL_VLOG(3,
              "HEXL_NTT_PARALLEL_DEPTH         = " << HEXL_NTT_PARALLEL_DEPTH);

    setup_done = true;

    // Add new threads if necessary
    if (n_threads > total_threads) {
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

    } else if (n_threads < total_threads) {
      size_t to_remove = total_threads - n_threads;
      for (size_t i = 0; i < to_remove; ++i) {
        // Kill thread
        ThreadInfo* thread_handler = thread_handlers.at(total_threads - 1);
        if (thread_handler->state.load() == STATE::SLEEPING) {
          thread_handler->state.store(STATE::KILL);
          thread_handler->waker.notify_one();
        } else {
          thread_handler->state.store(STATE::KILL);
        }
        thread_handler->thread.join();

        // Update handlers
        delete thread_handler;
        thread_handlers.pop_back();
        total_threads--;
      }
    }

    if (n_threads == 0) {
      setup_done = false;
    }

    HEXL_VLOG(2,
              "Setting up thread pool with " << total_threads << " threads.");
  }

 public:
  // Methods

  ThreadPool() { ThreadPool::child = false; }

  ~ThreadPool() { SetupThreads(0); }

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

      // Only if all threads are available
      if (next_thread.load() == 0) {
        next_thread.store(total_threads);

        for (size_t i = 0; i < total_threads; i++) {
          ThreadInfo* thread_handler = thread_handlers.at(i);
          switch (thread_handler->state.load()) {
            case STATE::DONE:
              thread_handler->task = job;
              thread_handler->state.store(STATE::KICK_OFF);
              break;
            case STATE::SLEEPING:
              thread_handler->task = job;
              thread_handler->state.store(STATE::KICK_OFF);
              thread_handler->waker.notify_one();
              break;
            default:  // In case thread is not on expected state
              job(i, total_threads);
          }
        }
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

  // AddRecursiveCalls: Runs a task on next thread available
  void AddRecursiveCalls(tp_task_t task_a, tp_task_t task_b) {
    HEXL_CHECK(task_a, "task_a: Require non empty task");
    HEXL_CHECK(task_b, "task_b: Require non empty task");
    bool locked = false;

    // Try using thread pool
    if (!ThreadPool::child) {
      locked = pool_mutex.try_lock();
      if (!locked) {
        task_a(0, 1);
        task_b(0, 1);
        return;
      }
    }

    if (!setup_done) {
      SetupThreads_Unlocked(HEXL_NUM_THREADS);  // Setup if thread pool down
    }

    size_t next = next_thread.fetch_add(2);
    if (next <= total_threads - 2) {
      ThreadInfo* thread_handler = thread_handlers.at(next++);
      switch (thread_handler->state.load()) {
        case STATE::DONE:
          thread_handler->task = task_a;
          thread_handler->state.store(STATE::KICK_OFF);
          break;
        case STATE::SLEEPING:
          thread_handler->task = task_a;
          thread_handler->state.store(STATE::KICK_OFF);
          thread_handler->waker.notify_one();
          break;
        default:  // In case thread is not on expected state
          task_a(next - 1, total_threads);
      }

      thread_handler = thread_handlers.at(next++);
      switch (thread_handler->state.load()) {
        case STATE::DONE:
          thread_handler->task = task_b;
          thread_handler->state.store(STATE::KICK_OFF);
          break;
        case STATE::SLEEPING:
          thread_handler->task = task_b;
          thread_handler->state.store(STATE::KICK_OFF);
          thread_handler->waker.notify_one();
          break;
        default:  // In case thread is not on expected state
          task_b(next - 1, total_threads);
      }

      // Implicit barrier
      for (size_t i = next - 2; i < next; i++) {
        WaitThread(thread_handlers.at(i));
      }

      // Next thread to be used
      next_thread.fetch_add(-2);

    } else {
      next_thread.fetch_add(-2);
      task_a(0, 1);
      task_b(0, 1);
    }

    if (locked) {
      pool_mutex.unlock();
    }
  }

  // SetupThreads: Spawns new threads if necessary
  void SetupThreads(size_t n_threads) {
    std::lock_guard<std::mutex> lock(pool_mutex);
    SetupThreads_Unlocked(n_threads);
  }

  // Return threads' handlers
  std::vector<const ThreadInfo*> GetThreadHandlers() {
    std::lock_guard<std::mutex> lock(pool_mutex);
    // returns a copy
    return std::vector<const ThreadInfo*>{thread_handlers.begin(),
                                          thread_handlers.end()};
  }
};

}  // namespace hexl
}  // namespace intel
