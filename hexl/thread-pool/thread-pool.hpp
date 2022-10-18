// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/util/check.hpp"
#include "thread-pool/thread-handler.hpp"

namespace intel {
namespace hexl {

class ThreadPool {
 public:
  // Methods

  ThreadPool() { isChildThread = false; }

  ~ThreadPool() { SetupThreads(0); }

  // GetNumThreads: Returns total number of threads
  size_t GetNumThreads() const {
    std::lock_guard<std::mutex> lock(pool_mutex);
    return thread_handlers.size();
  }

  // AddParallelJobs: Runs the same function on a total number of threads
  void AddParallelJobs(tp_task_t job) {
    HEXL_CHECK(job, "Require non empty job");

    // Try using thread pool
    if (pool_mutex.try_lock()) {
      if (!setup_done) {
        SetupThreads_Unlocked(
            HEXL_NUM_THREADS);  // Setup if thread pool is down
      }

      // Only if all threads are available
      if (next_thread.load() == 0) {
        const size_t t_threads = thread_handlers.size();
        next_thread.store(t_threads);

        for (size_t i = 0; i < t_threads; ++i) {
          auto handler = thread_handlers[i];
          switch (handler->state.load()) {
            case STATE::DONE:
              handler->task = job;
              handler->state.store(STATE::KICK_OFF);
              break;
            case STATE::SLEEPING:
              handler->task = job;
              handler->state.store(STATE::KICK_OFF);
              {
                std::lock_guard<std::mutex> lock(handler->wake_mutex);
                handler->waker.notify_one();
              }
              break;
            default:  // In case thread is not in expected state
              job(i, t_threads);
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
    if (!isChildThread) {
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

    const size_t t_threads = thread_handlers.size();
    size_t next = next_thread.fetch_add(2);
    if (next <= t_threads - 2) {
      ThreadHandler* handler = thread_handlers.at(next++);
      switch (handler->state.load()) {
        case STATE::DONE:
          handler->task = task_a;
          handler->state.store(STATE::KICK_OFF);
          break;
        case STATE::SLEEPING:
          handler->task = task_a;
          handler->state.store(STATE::KICK_OFF);
          {
            std::lock_guard<std::mutex> lock(handler->wake_mutex);
            handler->waker.notify_one();
          }
          break;
        default:  // In case thread is not on expected state
          task_a(next - 1, t_threads);
      }

      handler = thread_handlers.at(next++);
      switch (handler->state.load()) {
        case STATE::DONE:
          handler->task = task_b;
          handler->state.store(STATE::KICK_OFF);
          break;
        case STATE::SLEEPING:
          handler->task = task_b;
          handler->state.store(STATE::KICK_OFF);
          {
            std::lock_guard<std::mutex> lock(handler->wake_mutex);
            handler->waker.notify_one();
          }
          break;
        default:  // In case thread is not on expected state
          task_b(next - 1, t_threads);
      }

      // Implicit barrier
      for (size_t i = next - 2; i < next; i++) {
        WaitThread(thread_handlers.at(i));
      }

      // Next thread to be used
      next_thread.fetch_add(-2LL);

    } else {
      next_thread.fetch_add(-2LL);
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
  std::vector<const ThreadHandler*> GetThreadHandlers() const {
    std::lock_guard<std::mutex> lock(pool_mutex);
    // returns a copy
    return std::vector<const ThreadHandler*>{thread_handlers.begin(),
                                             thread_handlers.end()};
  }

 private:
  // Properties
  std::atomic_uint64_t next_thread{0};          // Points to next free thread
  std::vector<ThreadHandler*> thread_handlers;  // Thread's info
  mutable std::mutex pool_mutex;                // Control pool's edition
  bool setup_done = false;                      // Is thread pool initialized

  // Methods

  // StartThreads: Spawn a given number of threads
  void StartThreads(size_t new_threads) {
    // Add N new threads
    auto old_size = thread_handlers.size();
    thread_handlers.reserve(old_size + new_threads);
    for (size_t i = old_size; i < old_size + new_threads; ++i) {
      // Create handler and add it to vector of threads
      // Let handler know its parent container and its position in the vector
      thread_handlers.emplace_back(new ThreadHandler(thread_handlers, i));
    }
  }

  // WaitThread: Wait for one thread to be ready
  void WaitThread(ThreadHandler* thread_handler) {
    while (thread_handler->state.load() != STATE::DONE &&
           thread_handler->state.load() != STATE::SLEEPING) {
    }
  }

  // SetBarrier_Unlocked: Sets a barrier to sync all threads. Without mutex
  void SetBarrier_Unlocked() {
    // Waits until all threads are DONE or SLEEPING
    for (auto handler : thread_handlers) {
      WaitThread(handler);
    }

    // Next thread to be used
    next_thread.store(0);
  }

  // SetupThreads_Unlocked: Spawns new threads if necessary. Without mutex
  void SetupThreads_Unlocked(size_t n_threads) {
    if (n_threads == 0) {
      setup_done = false;
    } else {
      HEXL_VLOG(3, "Thread Pool Info:");
      HEXL_VLOG(3, "HEXL_NUM_THREADS        = " << HEXL_NUM_THREADS);
      HEXL_VLOG(3, "HEXL_NTT_PARALLEL_DEPTH = " << HEXL_NTT_PARALLEL_DEPTH);
      HEXL_VLOG(3, "HW Threads              = "
                       << std::thread::hardware_concurrency());
      setup_done = true;
    }

    // Add new threads if necessary
    size_t t_threads = thread_handlers.size();
    if (n_threads > t_threads) {
      // Can't exceed HW threads
      if (n_threads > std::thread::hardware_concurrency()) {
        n_threads = std::thread::hardware_concurrency();
        HEXL_VLOG(
            3, "Exceeded platform's available number of threads. Setting to: "
                   << std::thread::hardware_concurrency() << ".");
      }

      size_t new_threads = n_threads - t_threads;
      StartThreads(new_threads);  // Start extra threads

      // Wait for threads to be ready
      SetBarrier_Unlocked();

      // Remove threads if necessary
    } else if (n_threads < t_threads) {
      // Kill lambda
      auto kill_thread = [](ThreadHandler* thread_handler) {
        if (thread_handler->state.load() == STATE::SLEEPING) {
          thread_handler->state.store(STATE::KILL);
          thread_handler->waker.notify_one();
        } else {
          thread_handler->state.store(STATE::KILL);
        }
        thread_handler->thread.join();

        // Update handlers
        delete thread_handler;
      };

      std::for_each(thread_handlers.rbegin(),
                    thread_handlers.rend() - n_threads, kill_thread);
      thread_handlers.resize(n_threads);
    }
  }
};

}  // namespace hexl
}  // namespace intel
