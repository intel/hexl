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
#include <atomic>

namespace intel {
namespace hexl {

typedef struct s_thread_info {
  std::atomic_int state{3};
  std::atomic_flag flag;
  int64_t thread_id = 0;
  int64_t total_threads = 1;
  std::thread thread;
  //std::function<void(struct s_thread_info*)> task;
  std::function<void()> task;
} s_thread_info_t;

class ThreadPool {
 private:
  // ************ Properties ************
  int num_threads = 0;                        // Total number of threads
  std::vector<s_thread_info_t*> thread_handlers;  
  // Events lock and related variables
  bool stop_threads = false;

  // ************ Methods ************
  // Start a number of threads
  void StartThreads(int new_threads) {
    // std::cout << "ROCHA Start" << std::endl;
    for (int i = 0; i < new_threads; ++i) {
      s_thread_info_t* thread_handler = new s_thread_info_t();
      thread_handlers.emplace_back(thread_handler);

      thread_handler->thread = std::thread([thread_handler] {

        while (true) {
          {

            thread_handler->state.store(1);
          
            //std::cout << "ROCHA: Thread waiting. Thread ID " << thread_handler->thread_id << "." << std::endl;
            while (1) {
              if (thread_handler->state.load() == 2) break;
            }
            
            thread_handler->state.store(3);
            
            //std::cout << "ROCHA: Thread working. Thread ID " << thread_handler->thread_id << "." << std::endl;

          }
          //thread_handler->task(thread_handler);
          thread_handler->task();
        }
      });
    }
    num_threads += new_threads;
  }

  // Stop threads
  void StopThreads() {
    std::cout << "ROCHA Stop" << std::endl;
    {
      //std::lock_guard<std::mutex> lock{event_lock};
      stop_threads = true;
    }

    //wake_condition.notify_all();

    //for (auto& thread : threads) {
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
  //void AddJob(std::function<void(s_thread_info_t*)> task) {
  void AddJob(std::function<void()> task) {
    // std::cout << "ROCHA Added Job" << std::endl;
    
    for (int i = 0; i < num_threads; i++) {
      s_thread_info_t* thread_handler = thread_handlers.at(i);
      {
        thread_handler->task = task;
        //thread_handler->thread_id = i;
        //thread_handler->total_threads = num_threads;
        thread_handler->state.store(2);
        //std::cout << "ROCHA Added Job. ID: " << i << std::endl;
      }
    }
  }

  // Setup extra threads
  void SetupThreads(int n_threads) {
    // std::cout << "ROCHA Setup" << std::endl;
    if (num_threads < n_threads) {
      int new_threads = n_threads - num_threads;
      StartThreads(new_threads);
    }
  }

  // Get number of threads
  size_t GetNumThreads() { return num_threads; }

  //size_t GetLocalThreadId() { return thread_ids[std::this_thread::get_id()]; }

  void WaitThreads() {
    for(int i = 0; i < num_threads; i++){  
      s_thread_info_t* thread_handler = thread_handlers.at(i);  
      while (1) {
        if (thread_handler->state.load() == 1) {
          break;
        }
      }
    }
  }
};

class ThreadPoolExecutor {
 public:
  static void SetNumberOfThreads(int n_threads) {
    pool->SetupThreads(n_threads);
  }

  //static void AddParallelTask(std::function<void(s_thread_info_t*)> job) { 
  static void AddParallelTask(std::function<void()> job) { 
    pool->AddJob(job); 
  }

  static size_t GetNumberOfThreads() { return pool->GetNumThreads(); }

  //static size_t GetThreadId() { return pool->GetLocalThreadId(); }

  static void SetBarrier() { pool->WaitThreads(); }

 private:
  inline static ThreadPool* pool = new ThreadPool(1);
};

}  // namespace hexl
}  // namespace intel
