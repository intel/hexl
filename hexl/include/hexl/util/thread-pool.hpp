// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <iostream>
#include <map>
#include <queue>
#include <thread>
#include <vector>

namespace intel {
namespace hexl {

typedef struct s_thread_info {
  std::atomic_int state{3};
  std::atomic_flag flag;
  int64_t thread_id = 0;
  int64_t total_threads = 1;
  std::thread thread;
  // std::function<void(struct s_thread_info*)> task;
  // std::function<void()> task;
  std::function<void(int id, int threads)> task;
} s_thread_info_t;

class ThreadPool {
 private:
  // ************ Properties ************
  int num_threads = 0;  // Total number of threads
  std::atomic_int next_thread = 0;
  std::vector<s_thread_info_t*> thread_handlers;

  // Events lock and related variables
  bool stop_threads = false;

  // ************ Methods ************
  // Start a number of threads
  void StartThreads(int new_threads) {
    // std::cout << "ROCHA Start" << std::endl;
    int current_threads = num_threads;
    for (int i = 0; i < new_threads; ++i) {
      s_thread_info_t* thread_handler = new s_thread_info_t();
      thread_handler->state.store(0);
      thread_handlers.emplace_back(thread_handler);
      int* threads = &num_threads;
      thread_handler->thread =
          std::thread([thread_handler, current_threads, i, threads] {
            while (true) {
              // std::cout << "ROCHA: Thread waiting. Thread ID " <<
              // current_threads + i << std::endl;
              thread_handler->state.store(1);

              while (1) {
                if (thread_handler->state.load() == 2) break;
              }

              // std::cout << "ROCHA: Thread working. Thread ID " <<
              // current_threads + i << std::endl;
              thread_handler->state.store(3);

              thread_handler->task(current_threads + i, *threads);
            }
          });
    }
    num_threads += new_threads;
    // WaitThreads();
  }

  // Stop threads
  void StopThreads() {
    std::cout << "ROCHA Stop" << std::endl;
    {
      // std::lock_guard<std::mutex> lock{event_lock};
      stop_threads = true;
    }

    // wake_condition.notify_all();

    // for (auto& thread : threads) {
    //  thread.join();
    //}
  }

 public:
  // ************ Methods ************
  // Constructor
  explicit ThreadPool(int n_threads) {
    // std::cout << "ROCHA Thread pool constructed" << std::endl;
    std::cout << " struct " << sizeof(s_thread_info_t) << std::endl;
    StartThreads(n_threads);
  }

  // Destructor
  ~ThreadPool() {
    // std::cout << "ROCHA Thread pool destructor" << std::endl;
    StopThreads();
  }

  // Add jobs to the queue
  // void AddJob(std::function<void(s_thread_info_t*)> task) {
  // void AddJob(std::function<void()> task) {
  void AddParallelJobs(std::function<void(int id, int threads)> job) {
    // std::cout << "ROCHA Added Job" << std::endl;

    if (next_thread.load() == 0) {
      for (int i = 0; i < num_threads; i++) {
        s_thread_info_t* thread_handler = thread_handlers.at(i);
        {
          thread_handler->task = job;
          thread_handler->state.store(2);
          // std::cout << "ROCHA Added Job. ID: " << i << std::endl;
        }
      }
      next_thread.store(num_threads);
    } else {
      job(0, 1);
    }
  }

  void AddTask(std::function<void(int id, int threads)> task) {
    // std::cout << "ROCHA Added Job" << std::endl;
    int next = next_thread.fetch_add(1);
    if (next < num_threads) {
      // std::cout << "ROCHA Task Adding " << std::endl;
      s_thread_info_t* thread_handler = thread_handlers.at(next);
      thread_handler->task = task;
      thread_handler->state.store(2);
      // std::cout << "ROCHA Task Added " << available << std::endl;
    } else {
      // std::cout << "ROCHA No available " << std::endl;
      task(0, 1);
    }
  }
  // Setup extra threads
  void SetupThreads(int n_threads) {
    // std::cout << "ROCHA Setup" << std::endl;
    if (num_threads < n_threads) {
      // std::cout << "ROCHA Setup " << n_threads << " threads" << std::endl;
      int new_threads = n_threads - num_threads;
      StartThreads(new_threads);
    }
  }

  // Get number of threads
  size_t GetNumThreads() { return num_threads; }

  // size_t GetLocalThreadId() { return thread_ids[std::this_thread::get_id()];
  // }

  void WaitThreads() {
    // std::cout << "ROCHA Barrier on " << num_threads << " threads" <<
    // std::endl;
    for (int i = 0; i < num_threads; i++) {
      s_thread_info_t* thread_handler = thread_handlers.at(i);
      while (1) {
        if (thread_handler->state.load() == 1) {
          // std::cout << "ROCHA Thread "<< i << " Done" << std::endl;
          break;
        }
      }
    }
    next_thread.store(0);
    // std::cout << "ROCHA Barrier Done" << std::endl;
  }
};

class ThreadPoolExecutor {
 public:
  static void SetNumberOfThreads(int n_threads) {
    pool->SetupThreads(n_threads);
  }

  // static void AddParallelTask(std::function<void(s_thread_info_t*)> job) {
  // static void AddParallelTask(std::function<void()> job) {
  static void AddParallelTask(std::function<void(int id, int threads)> job) {
    pool->AddParallelJobs(job);
  }

  static void AddTask(std::function<void(int id, int threads)> job) {
    pool->AddTask(job);
  }

  static size_t GetNumberOfThreads() { return pool->GetNumThreads(); }

  // static size_t GetThreadId() { return pool->GetLocalThreadId(); }

  static void SetBarrier() { pool->WaitThreads(); }

 private:
  inline static ThreadPool* pool = new ThreadPool(1);
};

}  // namespace hexl
}  // namespace intel
