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
  int64_t thread_id = 0;
  int64_t total_threads = 1;
  std::thread thread;
  std::function<void(struct s_thread_info*)> task;
} s_thread_info_t;

class ThreadInfo{
public:
  ThreadInfo(){}
  ~ThreadInfo(){}

  int64_t state = 3;
  int64_t thread_id = 0;
  int64_t total_threads;
  std::thread thread;
  std::function<void(struct ThreadInfo*)> task;
  std::mutex event_lock;
  std::condition_variable wake_cond;

  void SetThreadID(uint8_t id) { thread_id = id; }
};

class ThreadPool {
 private:
  // ************ Properties ************
  int max_num_threads = 0;                    // Max number of threads
  int num_threads = 0;                        // Total number of threads
  std::vector<std::thread> threads;              // Threads Constainer
  std::vector<s_thread_info_t*> thread_handlers;  
  std::queue<std::function<void(int, int)>> jobs;        // Queue of tasks
  std::function<void(int, int)> parallel_job;
  std::map<std::thread::id, size_t> thread_ids;  // Local Ids
  std::atomic_int jobs_counter{0};
  std::atomic_int jobs_finished{0};
  // Events lock and related variables
  std::mutex event_lock;
  std::condition_variable wake_condition;
  std::condition_variable finish_condition;
  size_t working_threads = 0;
  bool stop_threads = false;
  bool start_work = false;

  // ************ Methods ************
  // Start a number of threads
  void StartThreads(int new_threads) {
    // std::cout << "ROCHA Start" << std::endl;
    working_threads += new_threads;
    int start = num_threads;
    for (int i = 0; i < new_threads; ++i) {
      s_thread_info_t* thread_handler = new s_thread_info_t();
      thread_handlers.emplace_back(thread_handler);
      thread_handler->thread_id = start + i;
      thread_handler->thread = std::thread([=] {

        while (true) {
          // std::function<void(int, int)> Job;
          //std::function<void(ThreadInfo*)> Job;
          {
            //std::unique_lock<std::mutex> t_wake_lock{event_lock};
            // std::unique_lock<std::mutex> t_wake_lock{thread_handler->event_lock};
            // working_threads--;
            // if (working_threads == 0) {
            //   t_wake_lock.unlock();
            //  start_work = false;
            //  finish_condition.notify_one();
            //  t_wake_lock.lock();
            //}
            //std::cout << "ROCHA Thread " << start + i << " to wait. Working: " << working_threads << std::endl;
            //wake_condition.wait(t_wake_lock,
            //                    [=] { return stop_threads || !jobs.empty(); });
            // 
            
            thread_handler->state.store(1);
            

            //std::cout << "ROCHA: Thread waiting. Thread ID " << thread_handler->thread_id << "." << std::endl;
            while (1) {
              if (thread_handler->state.load() == 2) break;
            }
            
            thread_handler->state.store(3);
            
            //std::cout << "ROCHA: Thread working. Thread ID " << thread_handler->thread_id << "." << std::endl;

            //thread_handler->wake_cond.wait(t_wake_lock,
            //                    [=] { return thread_handler->state == 2; });
            // std::cout << "ROCHA: Thread working. Thread ID " << thread_handler->thread_id << "." << std::endl;
            //thread_handler->state = 3;
            // if (stop_threads) break;

            //std::cout << "ROCHA Thread " << start + i << " to work. Working: " << working_threads << std::endl;

            //Job = std::move(jobs.front());
            //jobs.pop();
          }
          // parallel_job(start + i, 32);
          //Job(start + i, 2);
          thread_handler->task(thread_handler);
        }
      });
      // std::cout << "ID " << std::this_thread::get_id() << " [" << start + i
      //          << "]" << std::endl;
      // thread_ids.insert({threads.at(start + i).get_id(), start + i});
    }
    num_threads += new_threads;
  }

  // Stop threads
  void StopThreads() {
    std::cout << "ROCHA Stop" << std::endl;
    {
      std::lock_guard<std::mutex> lock{event_lock};
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
  explicit ThreadPool(int n_threads) {
    // std::cout << "ROCHA Thread pool constructed" << std::endl;
    std::cout << " struct " << sizeof(s_thread_info_t) << std::endl;
    max_num_threads = n_threads;
    StartThreads(n_threads);
  }

  // Destructor
  ~ThreadPool() {
    // std::cout << "ROCHA Thread pool destructor" << std::endl;
    StopThreads();
  }

  // Add jobs to the queue
  // void AddJob(std::function<void(int, int)> job) {
  void AddJob(std::function<void(s_thread_info_t*)> task) {
    // std::cout << "ROCHA Added Job" << std::endl;
    
  //{
    /*
    std::lock_guard<std::mutex> lock{event_lock};
    working_threads = num_threads;
    for (size_t i = 0; i < num_threads; i++) {
      // std::cout << "ROCHA Adding Job. Working: " << working_threads <<
      // std::endl;

      start_work = true;
      jobs.emplace(job);  // Why to use move?
      wake_condition.notify_one();
    }
    */
    /*
    std::lock_guard<std::mutex> lock{event_lock};
    start_work = true;
    working_threads = num_threads;
    parallel_job = job;  // Why to use move?
    std::cout << "ROCHA Adding Jobs. Working: " << working_threads << std::endl;
    wake_condition.notify_all();
    */
  //}
    // 
    for (int i = 0; i < num_threads; i++) {
      s_thread_info_t* thread_handler = thread_handlers.at(i);
      {
        //std::unique_lock<std::mutex> lock{thread_handler->event_lock};
        thread_handler->task = task;
        //thread_handler->state = 2;
        thread_handler->thread_id = i;
        thread_handler->total_threads = num_threads;
        thread_handler->state.store(2);
        //std::cout << "ROCHA Added Job. ID: " << i << std::endl;
        //lock.unlock();
        //thread_handler->wake_cond.notify_one();
      }
    }
  }

  // Setup extra threads
  void SetupThreads(int n_threads) {
    // std::cout << "ROCHA Setup" << std::endl;
    if (num_threads < n_threads) {
      int new_threads = n_threads - num_threads;
      max_num_threads = n_threads;
      StartThreads(new_threads);
    }
  }

  // Get number of threads
  size_t GetNumThreads() { return num_threads; }

  size_t GetLocalThreadId() { return thread_ids[std::this_thread::get_id()]; }

  void WaitThreads() {
    bool all_complete = false;
    
    while (!all_complete) {
      all_complete = true;
      for(int i = 0; i < num_threads; i++){
        s_thread_info_t* thread_handler = thread_handlers.at(i);
        {
          //std::unique_lock<std::mutex> lock{thread_handler->event_lock};
          if (thread_handler->state.load() != 1) {
            all_complete = false;
            break;
          }
        }
      }
    }
    // while (1){
    //   if (jobs_finished == jobs_counter) break;
    // }
    // jobs_counter = 0;
    // jobs_finished = 0;
    // std::cout << "ROCHA All Jobs finished." << jobs_counter << std::endl;
    // std::unique_lock<std::mutex> lock{event_lock};
    // finish_condition.wait(lock, [=] { return working_threads == 0; });
    // std::cout << "ROCHA All Jobs finished. Working: " << working_threads <<
    // std::endl;
  }
};

class ThreadPoolExecutor {
 public:
  static void SetNumberOfThreads(int n_threads) {
    pool->SetupThreads(n_threads);
  }

  //static void AddParallelTask(std::function<void(int, int)> job) { pool->AddJob(job); }
  static void AddParallelTask(std::function<void(s_thread_info_t*)> job) { pool->AddJob(job); }

  static size_t GetNumberOfThreads() { return pool->GetNumThreads(); }

  static size_t GetThreadId() { return pool->GetLocalThreadId(); }

  static void SetBarrier() { pool->WaitThreads(); }

 private:
  inline static ThreadPool* pool = new ThreadPool(1);
};
}  // namespace hexl
}  // namespace intel
