// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <list>

#include "test/test-util.hpp"
#include "thread-pool/thread-pool-executor.hpp"
#include "thread-pool/thread-pool-vars-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_MULTI_THREADING

// Testing number of threads across different phases
TEST(ThreadPool, GetNumberOfThreads) {
  // With setup. Correspond to SetNumberOfThreads.
  uint64_t nthreads = 2;
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());
  ThreadPoolExecutor::SetNumberOfThreads(0);

  // When running parallel jobs. Without previous setup.
  HEXL_NUM_THREADS = 2;
  ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  });
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());
  ThreadPoolExecutor::SetNumberOfThreads(0);

  // When running parallel task. Without previous setup.
  ThreadPoolExecutor::AddRecursiveCalls(
      [](int id, int threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      },
      [](int id, int threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      });
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  // When sleeping. Keep the same value.
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  // Try at 80% of available threads
  nthreads = static_cast<uint64_t>(std::thread::hardware_concurrency() * 0.7);
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  // When stopped. Returns zero.
  ThreadPoolExecutor::SetNumberOfThreads(0);
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), 0);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());
}

// Testing function that sets number of threads from env variable
TEST(ThreadPool, setup_num_threads_env_var) {
  // Max or default value result
  auto max_or_default = std::min<uint64_t>(HEXL_DEFAULT_NUM_THREADS,
                                           std::thread::hardware_concurrency());

  // Overshooting: Max HW's value is set
  {
    char env[] = "HEXL_NUM_THREADS=999999999";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, std::thread::hardware_concurrency());
  }

  // Wanted value is set
  {
    char env[] = "HEXL_NUM_THREADS=2";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, 2);
  }

  // Floating point: Rounded value is set
  {
    char env[] = "HEXL_NUM_THREADS=1.5";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, 1);
  }

  // Undefined: Default value is set
  {
    char env[] = "HEXL_NUM_THREADS";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, max_or_default);
  }

  // Negative: Default value is set
  {
    char env[] = "HEXL_NUM_THREADS=-2";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, max_or_default);
  }

  // String: Default value is set
  {
    char env[] = "HEXL_NUM_THREADS=error";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, max_or_default);
  }

  // Env var is used for thread pool size
  // Set to different than default or previously set value
  {
    HEXL_NUM_THREADS = 1;
    ThreadPoolExecutor::SetNumberOfThreads(0);
    ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
    });
    auto value = ThreadPoolExecutor::GetThreadHandlers();
    ASSERT_EQ(value.size(), 1);
  }

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Testing function that sets number of parallel ntt calls from env variable
TEST(ThreadPool, setup_ntt_calls_env_var) {
  HEXL_NUM_THREADS = 2;

  // Wanted value is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH=1";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, 1);
  }

  // Overshooting HEXL_NTT_PARALLEL_DEPTH: Zero is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH=999999999";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, 0);
  }

  // Undefined: Default value is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, HEXL_DEFAULT_NTT_PARALLEL_DEPTH);
  }

  // Negative: Default value is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH=-2";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, HEXL_DEFAULT_NTT_PARALLEL_DEPTH);
  }

  // Floating point: Rounded value is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH=1.5";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, 1);
  }

  // String: Default value is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH=error";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, HEXL_DEFAULT_NTT_PARALLEL_DEPTH);
  }
}

// Test setting number of threads programmatically
TEST(ThreadPool, SetNumberOfThreads) {
  // Overshooting HEXL_NUM_THREADS: Max HW's value is set
  {
    ThreadPoolExecutor::SetNumberOfThreads(999999999);
    auto value = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(value, std::thread::hardware_concurrency());
  }

  {
    // Precedence over env variable
    uint64_t nthreads = 2;
    HEXL_NUM_THREADS = 1;
    ThreadPoolExecutor::SetNumberOfThreads(nthreads);
    auto value = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(value, nthreads);

    // Setting new bigger value
    nthreads = 2;
    ThreadPoolExecutor::SetNumberOfThreads(nthreads);
    value = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(value, nthreads);

    // Setting new smaller value.
    nthreads = 0;
    ThreadPoolExecutor::SetNumberOfThreads(nthreads);
    value = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(value, nthreads);
  }

  {
    // Test: N threads get started
    int counter = 0;
    ThreadPoolExecutor::SetNumberOfThreads(2);
    auto handlers = ThreadPoolExecutor::GetThreadHandlers();
    for (size_t i = 0; i < handlers.size(); i++) {
      auto handler = handlers.at(i);
      if (handler->state.load() == STATE::DONE ||
          handler->state.load() == STATE::SLEEPING) {
        counter++;
      }
    }
    ASSERT_EQ(counter, ThreadPoolExecutor::GetNumberOfThreads());

    // Test: N threads get to sleep
    // Wait for threads to sleep
    std::this_thread::sleep_for(
        std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));

    counter = 0;
    for (size_t i = 0; i < handlers.size(); i++) {
      auto handler = handlers.at(i);
      if (handler->state.load() == STATE::SLEEPING) {
        counter++;
      }
    }
    ASSERT_EQ(counter, ThreadPoolExecutor::GetNumberOfThreads());
  }

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test synchronization barrier
TEST(ThreadPool, ImplicitBrriers) {
  int delay = 2;
  ThreadPoolExecutor::SetNumberOfThreads(2);  // Implicit barrier

  // Barrier waits 'til threads are done after parallel jobs
  tp_task_t task([=](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  });

  auto start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddParallelJobs(task);
  auto end = std::chrono::steady_clock::now();

  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, delay);

  // After parallel tasks
  start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddRecursiveCalls(task, task);
  end = std::chrono::steady_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count();
  ASSERT_GE(duration, delay);

  // One thread is sleeping
  start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddRecursiveCalls(
      [](size_t id, size_t threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
      },
      [](size_t id, size_t threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
      });
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count();
  ASSERT_GE(duration, 2 * HEXL_THREAD_WAIT_TIME);
  // Barrier work on sleeping threads

  // On nested tasks
  ThreadPoolExecutor::SetNumberOfThreads(6);  // Implicit barrier
  start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddRecursiveCalls(
      [=](size_t id, size_t threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        ThreadPoolExecutor::AddRecursiveCalls(task, task);
      },
      [=](size_t id, size_t threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        ThreadPoolExecutor::AddRecursiveCalls(task, task);
      });
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count();
  ASSERT_GE(duration, delay);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test threads are deleted
TEST(ThreadPool, StopThreads) {
  // Stop waiting threads
  ThreadPoolExecutor::SetNumberOfThreads(2);
  ThreadPoolExecutor::SetNumberOfThreads(0);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);

  // Stop done threads after parallel jobs
  ThreadPoolExecutor::SetNumberOfThreads(2);
  ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  });
  ThreadPoolExecutor::SetNumberOfThreads(0);  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);

  // Stop done threads after parallel tasks
  ThreadPoolExecutor::SetNumberOfThreads(2);
  ThreadPoolExecutor::AddRecursiveCalls(
      [](int id, int threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      },
      [](int id, int threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      });
  ThreadPoolExecutor::SetNumberOfThreads(0);  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);

  // Stop sleeping threads
  ThreadPoolExecutor::SetNumberOfThreads(2);
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::SetNumberOfThreads(0);  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);
}

// Test adding parallel loop jobs
TEST(ThreadPool, AddParallelJob) {
  std::list<std::thread::id> ids;
  std::mutex list_mutex;
  HEXL_NUM_THREADS = 4;

  // Common task
  tp_task_t id_task([&list_mutex, &ids](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::lock_guard<std::mutex> lock(list_mutex);
    ids.push_back(std::this_thread::get_id());
  });

  // Test: Add jobs without previous setup
  ThreadPoolExecutor::AddParallelJobs(id_task);
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ids.clear();

  // Test: Add jobs with previous setup different than env variable
  ThreadPoolExecutor::SetNumberOfThreads(8);
  ThreadPoolExecutor::AddParallelJobs(id_task);
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ids.clear();

  // Test: Add jobs on same thread pool when previous jobs are done
  ThreadPoolExecutor::AddParallelJobs(id_task);
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ids.clear();

  // Test: Add jobs when threads are sleeping
  // Wait for threads from previous test to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::AddParallelJobs(id_task);
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ids.clear();

  // Test: Add jobs when threads are busy
  ThreadPoolExecutor::SetNumberOfThreads(0);
  std::thread thread_object([]() {
    ThreadPoolExecutor::SetNumberOfThreads(2);
    ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::this_thread::sleep_for(std::chrono::milliseconds(8));
    });
  });

  // Give time for previous threads to be running
  std::this_thread::sleep_for(std::chrono::milliseconds(4));

  ThreadPoolExecutor::AddParallelJobs(id_task);

  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), 1);
  thread_object.join();

  ids.clear();

  // Test: Testing id and threads parameters
  std::list<int> expected, result;
  int nthreads = ThreadPoolExecutor::GetNumberOfThreads();
  for (int i = 0; i < nthreads; i++) {
    expected.push_back(i);
    expected.push_back(nthreads);
  }
  ThreadPoolExecutor::AddParallelJobs(
      [&list_mutex, &result](int id, int threads) {
        std::lock_guard<std::mutex> lock(list_mutex);
        result.push_back(id);
        result.push_back(threads);
      });
  expected.sort();
  result.sort();
  ASSERT_EQ(expected, result);

  // Test: 80% of available threads
  nthreads = static_cast<uint64_t>(std::thread::hardware_concurrency() * 0.7);
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddParallelJobs(id_task);
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test adding parallel tasks
TEST(ThreadPool, AddRecursiveCalls_1) {
  std::list<std::thread::id> ids;
  std::mutex list_mutex;
  HEXL_NUM_THREADS = 4;
  int delay = 2;

  // Common tasks
  tp_task_t sleep_task([delay](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  });

  tp_task_t id_task([&list_mutex, &ids](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::lock_guard<std::mutex> lock(list_mutex);
    ids.push_back(std::this_thread::get_id());
  });

  ThreadPoolExecutor::SetNumberOfThreads(0);

  // Test: Add tasks without previous setup.
  ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), 2);
  ids.clear();

  // Test: Add tasks with previous setup different than env var.
  uint64_t nthreads = 2;
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);  // Setup
  ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), nthreads);
  ids.clear();

  // Test: Add tasks on same thread pool when previous jobs (Prev test) are done
  // Using more than available threads
  ThreadPoolExecutor::AddRecursiveCalls(
      [&](int id, int threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
        std::lock_guard<std::mutex> lock(list_mutex);
        ids.push_back(std::this_thread::get_id());
      },
      [&](int id, int threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
        std::lock_guard<std::mutex> lock(list_mutex);
        ids.push_back(std::this_thread::get_id());
      });

  ids.sort();
  ASSERT_EQ(ids.size(), 6);  // calls
  ids.unique();
  ASSERT_EQ(ids.size(), nthreads);  // threads
  ids.clear();

  // Test: Add tasks when threads are sleeping. Threads from previous test.
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), 2);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test adding parallel tasks
TEST(ThreadPool, AddRecursiveCalls_2) {
  if (std::thread::hardware_concurrency() < 4) {
    GTEST_SKIP();
  }

  std::list<std::thread::id> ids;
  std::mutex list_mutex;
  HEXL_NUM_THREADS = 4;
  int delay = 2;

  // Common tasks
  tp_task_t sleep_task([delay](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  });

  tp_task_t id_task([&list_mutex, &ids](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::lock_guard<std::mutex> lock(list_mutex);
    ids.push_back(std::this_thread::get_id());
  });

  // Test: Add jobs when threads are busy without more free threads
  ThreadPoolExecutor::SetNumberOfThreads(2);
  std::thread thread_object(
      [=]() { ThreadPoolExecutor::AddRecursiveCalls(sleep_task, sleep_task); });
  // Give time for previous threads to be running
  std::this_thread::sleep_for(std::chrono::milliseconds(delay / 2));
  ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);

  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), 1);
  thread_object.join();
  ids.clear();

  // Test: Add jobs when threads are busy with more free threads
  uint64_t nthreads = 4;
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  std::thread thread_object2(
      [=] { ThreadPoolExecutor::AddRecursiveCalls(sleep_task, sleep_task); });

  // Give time for previous threads to be running
  std::this_thread::sleep_for(std::chrono::milliseconds(delay / 2));
  ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);

  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), 1);
  thread_object2.join();
  ids.clear();

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test thread safety of threa pool
TEST(ThreadPool, thread_safety) {
  std::atomic_int sync{2};
  int N_size = 100;
  std::atomic_int tasks_counter{0};
  std::atomic_int iterations{0};

  // Common tasks
  tp_task_t add_task([&tasks_counter](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    tasks_counter.fetch_add(1);
  });
  tp_task_t add_iterations([&](int id, int threads) {
    HEXL_UNUSED(id);
    iterations.fetch_add(N_size / threads);
  });

  // Parallel setups
  {
    sync.store(2);
    ThreadPoolExecutor::SetNumberOfThreads(0);
    std::thread thread_object1([&sync]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::SetNumberOfThreads(2);
    });
    std::thread thread_object2([&sync]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::SetNumberOfThreads(4);
    });
    thread_object1.join();
    thread_object2.join();
    uint64_t nthreads = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_TRUE(nthreads == 4 || nthreads == 2 ||
                nthreads == std::thread::hardware_concurrency());
  }

  //  Parallel stops
  {
    sync.store(2);
    ThreadPoolExecutor::SetNumberOfThreads(4);
    std::thread thread_object1([&sync]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::SetNumberOfThreads(0);
    });
    std::thread thread_object2([&sync]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::SetNumberOfThreads(0);
    });
    thread_object1.join();
    thread_object2.join();
    uint64_t nthreads = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(nthreads, 0);
  }

  // Setup and stop in parallel
  {
    sync.store(2);
    ThreadPoolExecutor::SetNumberOfThreads(0);
    std::thread thread_object1([&sync]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::SetNumberOfThreads(4);
    });
    std::thread thread_object2([&sync]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::SetNumberOfThreads(0);
    });
    thread_object1.join();
    thread_object2.join();
    uint64_t nthreads = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_TRUE(nthreads == 0 || nthreads == 4 ||
                nthreads == std::thread::hardware_concurrency());
  }

  // Parallel add task
  {
    sync.store(2);
    tasks_counter.store(0);
    ThreadPoolExecutor::SetNumberOfThreads(4);
    std::thread thread_object1([&]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::AddRecursiveCalls(add_task, add_task);
    });
    std::thread thread_object2([&]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::AddRecursiveCalls(add_task, add_task);
    });

    thread_object1.join();
    thread_object2.join();

    int counter = 0;
    auto handlers = ThreadPoolExecutor::GetThreadHandlers();
    for (size_t i = 0; i < handlers.size(); i++) {
      auto handler = handlers.at(i);
      if (handler->state.load() == STATE::DONE) {
        counter++;
      }
    }
    // Account for thread pools of 3 threads as tasks are added in pairs
    ASSERT_NEAR(counter, ThreadPoolExecutor::GetNumberOfThreads(), 1);
    ASSERT_EQ(tasks_counter.load(), 4);
  }

  // Nested add task
  {
    sync.store(2);
    tasks_counter.store(0);
    ThreadPoolExecutor::SetNumberOfThreads(6);

    ThreadPoolExecutor::AddRecursiveCalls(
        [&](int id, int threads) {
          HEXL_UNUSED(id);
          HEXL_UNUSED(threads);
          tasks_counter.fetch_add(1);
          ThreadPoolExecutor::AddRecursiveCalls(add_task, add_task);
        },
        [&](int id, int threads) {
          HEXL_UNUSED(id);
          HEXL_UNUSED(threads);
          tasks_counter.fetch_add(1);
          ThreadPoolExecutor::AddRecursiveCalls(add_task, add_task);
        });

    int counter = 0;
    auto handlers = ThreadPoolExecutor::GetThreadHandlers();
    for (size_t i = 0; i < handlers.size(); i++) {
      auto handler = handlers.at(i);
      if (handler->state.load() == STATE::DONE) {
        counter++;
      }
    }
    // Account for thread pools of 3 or 5 threads as tasks are added in pairs
    ASSERT_NEAR(counter, ThreadPoolExecutor::GetNumberOfThreads(), 1);
    ASSERT_EQ(tasks_counter.load(), 6);
  }

  // Add task & stop threads in parallel
  {
    sync.store(2);
    tasks_counter.store(0);
    ThreadPoolExecutor::SetNumberOfThreads(0);
    std::thread thread_object1([&]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::AddRecursiveCalls(add_task, add_task);
    });
    std::thread thread_object2([&sync]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::SetNumberOfThreads(0);
    });

    thread_object1.join();
    thread_object2.join();

    uint64_t pool_size = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_TRUE(pool_size == 0 || pool_size == HEXL_NUM_THREADS ||
                pool_size == std::thread::hardware_concurrency());
    ASSERT_EQ(tasks_counter.load(), 2);
  }

  // Parallel add jobs
  {
    sync.store(2);
    iterations.store(0);
    int N_size = 100;
    ThreadPoolExecutor::SetNumberOfThreads(4);
    std::thread thread_object1([&]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::AddParallelJobs(add_iterations);
    });
    std::thread thread_object2([&]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::AddParallelJobs(add_iterations);
    });

    thread_object1.join();
    thread_object2.join();

    int counter = 0;
    auto handlers = ThreadPoolExecutor::GetThreadHandlers();
    for (size_t i = 0; i < handlers.size(); i++) {
      auto handler = handlers.at(i);
      if (handler->state.load() == STATE::DONE) {
        counter++;
      }
    }
    ASSERT_EQ(counter, ThreadPoolExecutor::GetNumberOfThreads());
    ASSERT_EQ(iterations.load(), 2 * N_size);
  }

  // Add jobs and setup threads in parallel
  {
    sync.store(2);
    iterations.store(0);
    int nthreads = 4;
    ThreadPoolExecutor::SetNumberOfThreads(nthreads);
    std::thread thread_object1([&]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::AddParallelJobs(add_iterations);
    });
    std::thread thread_object2([&sync, nthreads]() {
      sync.fetch_add(-1);
      while (sync) {
      }
      ThreadPoolExecutor::SetNumberOfThreads(nthreads + 2);
    });

    thread_object1.join();
    thread_object2.join();

    int counter = 0;
    auto handlers = ThreadPoolExecutor::GetThreadHandlers();
    for (size_t i = 0; i < handlers.size(); i++) {
      auto handler = handlers.at(i);
      if (handler->state.load() == STATE::DONE) {
        counter++;
      }
    }
    ASSERT_EQ(counter, ThreadPoolExecutor::GetNumberOfThreads());
    ASSERT_EQ(iterations.load(), N_size);
  }
}

#ifdef HEXL_DEBUG
// Testing debug features
TEST(ThreadPool, bad_input) {
  tp_task_t task([](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
  });

  EXPECT_ANY_THROW(ThreadPoolExecutor::AddParallelJobs(nullptr));

  EXPECT_ANY_THROW(ThreadPoolExecutor::AddRecursiveCalls(nullptr, task));

  EXPECT_ANY_THROW(ThreadPoolExecutor::AddRecursiveCalls(task, nullptr));
}
#endif

#endif
}  // namespace hexl
}  // namespace intel
