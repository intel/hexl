// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

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

class ThreadPool {
 private:
  // ************ Properties ************
  size_t max_num_threads = 0;                    // Max number of threads
  size_t num_threads = 0;                        // Total number of threads
  std::vector<std::thread> threads;              // Threads Constainer
  std::queue<std::function<void()>> jobs;        // Queue of tasks
  std::map<std::thread::id, size_t> thread_ids;  // Local Ids

  // Events lock and related variables
  std::mutex event_lock;
  std::condition_variable wake_condition;
  std::condition_variable finish_condition;
  size_t working_threads = 0;
  bool stop_threads = false;

  // ************ Methods ************
  // Start a number of threads
  void StartThreads(size_t new_threads) {
    // std::cout << "ROCHA Start" << std::endl;
    working_threads += new_threads;
    size_t start = num_threads;
    for (size_t i = 0; i < new_threads; ++i) {
      threads.emplace_back([=] {
        while (true) {
          std::function<void()> Job;
          {
            std::unique_lock<std::mutex> t_wake_lock{event_lock};
            working_threads--;
            if (working_threads == 0) {
              t_wake_lock.unlock();
              finish_condition.notify_one();
              t_wake_lock.lock();
            }
            // std::cout << "ROCHA Thread " << GetLocalThreadId() << " to wait.
            // Working: " << working_threads << std::endl;
            wake_condition.wait(t_wake_lock,
                                [=] { return stop_threads || !jobs.empty(); });

            if (stop_threads) break;

            // std::cout << "ROCHA Thread " << GetLocalThreadId() << " to work.
            // Working: " << working_threads << std::endl;

            Job = std::move(jobs.front());
            jobs.pop();
          }
          Job();
        }
      });
      std::cout << "ID " << std::this_thread::get_id() << " [" << start + i
                << "]" << std::endl;
      thread_ids.insert({threads.at(start + i).get_id(), start + i});
      num_threads++;
    }
  }

  // Stop threads
  void StopThreads() {
    std::cout << "ROCHA Stop" << std::endl;
    {
      std::unique_lock<std::mutex> lock{event_lock};
      stop_threads = true;
    }

    wake_condition.notify_all();

    for (auto& thread : threads) {
      thread.join();
    }
  }

 public:
  // ************ Methods ************
  // Constructor
  explicit ThreadPool(size_t n_threads) {
    // std::cout << "ROCHA Thread pool constructed" << std::endl;
    max_num_threads = n_threads;
    StartThreads(n_threads);
  }

  // Destructor
  ~ThreadPool() {
    // std::cout << "ROCHA Thread pool destructor" << std::endl;
    StopThreads();
  }

  // Add jobs to the queue
  void AddJob(std::function<void()> job) {
    // std::cout << "ROCHA Added Job" << std::endl;
    {
      for (size_t i = 0; i < num_threads; i++) {
        std::unique_lock<std::mutex> lock{event_lock};
        working_threads++;
        // std::cout << "ROCHA Adding Job. Working: " << working_threads <<
        // std::endl;
        jobs.emplace(job);  // Why to use move?
        wake_condition.notify_one();
      }
    }
  }

  // Setup extra threads
  void SetupThreads(size_t n_threads) {
    // std::cout << "ROCHA Setup" << std::endl;
    if (num_threads < n_threads) {
      size_t new_threads = n_threads - num_threads;
      max_num_threads = n_threads;
      StartThreads(new_threads);
    }
  }

  // Get number of threads
  size_t GetNumThreads() { return num_threads; }

  size_t GetLocalThreadId() { return thread_ids[std::this_thread::get_id()]; }

  void WaitThreads() {
    std::unique_lock<std::mutex> lock{event_lock};
    finish_condition.wait(lock, [=] { return working_threads == 0; });
    // std::cout << "ROCHA All Jobs finished. Working: " << working_threads <<
    // std::endl;
  }
};

class ThreadPoolExecutor {
 public:
  static void SetNumberOfThreads(size_t n_threads) {
    pool->SetupThreads(n_threads);
  }

  static void AddParallelTask(std::function<void()> job) { pool->AddJob(job); }

  static size_t GetNumberOfThreads() { return pool->GetNumThreads(); }

  static size_t GetThreadId() { return pool->GetLocalThreadId(); }

  static void SetBarrier() { pool->WaitThreads(); }

 private:
  inline static ThreadPool* pool = new ThreadPool(2);
};
}  // namespace hexl
}  // namespace intel
