// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "test/test-thread-pool-common.hpp"
#include "test/test-thread-pool-util.hpp"
#include "test/test-util.hpp"
#include "thread-pool/thread-pool-vars-util.hpp"

#ifdef HEXL_MULTI_THREADING

namespace intel {
namespace hexl {

// Env Variables ***************************************************************

// Testing function that sets number of threads from env variable
TEST(ThreadPool, setup_num_threads_env_var) {
  // Max or default value result
  auto max_or_default = std::min<uint64_t>(HEXL_DEFAULT_NUM_THREADS,
                                           std::thread::hardware_concurrency());

  // Overshooting: Max HW's value is set
  {
    char env[] = "HEXL_NUM_THREADS=999999";
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

  // Floating point: Rounded value is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH=1.5";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, 1);
  }
}

// Testing number of threads across different phases ***************************

// After setup. Correspond to SetNumberOfThreads.
TEST_P(ParallelThreads, GetNumberOfThreads_after_setup) {
  uint64_t nthreads = GetParam();
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }

  ThreadPoolExecutor::SetNumberOfThreads(0);

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// After stopped. Returns zero.
TEST_P(ParallelThreads, GetNumberOfThreads_after_stop) {
  uint64_t nthreads = GetParam();
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  ThreadPoolExecutor::SetNumberOfThreads(0);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), 0);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());
}

// After running parallel jobs. Without previous setup.
TEST(ThreadPool, GetNumberOfThreads_after_AddParallelJobs) {
  uint64_t nthreads = 2;
  int N_size = 100;
  ThreadPoolExecutor::SetNumberOfThreads(0);

  HEXL_NUM_THREADS = nthreads;
  ThreadPoolExecutor::AddParallelJobs(N_size, dummy_task);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// After running AddRecursiveCalls. Without previous setup.
TEST(ThreadPool, GetNumberOfThreads_after_AddRecursiveCalls) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(0);

  HEXL_NUM_THREADS = nthreads;
  ThreadPoolExecutor::AddRecursiveCalls(0, 0, dummy_task, dummy_task);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// After sleeping. Keep the same value.
TEST(ThreadPool, GetNumberOfThreads_after_sleeping) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test setting number of threads programmatically *****************************

// Overshooting HEXL_NUM_THREADS: Max HW's value is set
TEST(ThreadPool, SetNumberOfThreads_overshoot) {
  ThreadPoolExecutor::SetNumberOfThreads(999999999);
  auto value = ThreadPoolExecutor::GetNumberOfThreads();
  ASSERT_EQ(value, std::thread::hardware_concurrency());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Precedence over env variable
TEST(ThreadPool, SetNumberOfThreads_precedence) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(0);

  HEXL_NUM_THREADS = nthreads >> 1;
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  auto value = ThreadPoolExecutor::GetNumberOfThreads();
  ASSERT_EQ(value, nthreads);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: N threads get started
TEST_P(ParallelThreads, SetNumberOfThreads_state_setup) {
  uint64_t nthreads = GetParam();
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }
  uint64_t counter = 0;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  for (auto handler : handlers) {
    if (handler->state.load() == STATE::DONE ||
        handler->state.load() == STATE::SLEEPING) {
      counter++;
    }
  }
  ASSERT_EQ(counter, ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Setting new bigger value
TEST(ThreadPool, SetNumberOfThreads_set_bigger_value) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads >> 1);

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  auto value = ThreadPoolExecutor::GetNumberOfThreads();
  ASSERT_EQ(value, nthreads);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Setting new smaller value.
TEST(ThreadPool, SetNumberOfThreads_set_smaller_value) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  ThreadPoolExecutor::SetNumberOfThreads(nthreads >> 1);
  auto value = ThreadPoolExecutor::GetNumberOfThreads();
  ASSERT_EQ(value, nthreads >> 1);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: N threads get to sleep
TEST_P(ParallelThreads, SetNumberOfThreads_state_sleeping) {
  uint64_t nthreads = GetParam();
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }
  uint64_t counter = 0;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  for (auto handler : handlers) {
    if (handler->state.load() == STATE::SLEEPING) {
      counter++;
    }
  }
  ASSERT_EQ(counter, ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// StopThreads *****************************************************************

// Stop done threads after parallel jobs
TEST(ThreadPool, StopThreads_after_AddParallelJobs) {
  uint64_t nthreads = 2;
  int N_size = 100;
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddParallelJobs(N_size, working_task);
  ThreadPoolExecutor::SetNumberOfThreads(0);  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);
}

// Stop done threads after recursive tasks
TEST(ThreadPool, StopThreads_after_AddRecursiveCalls) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddRecursiveCalls(0, 0, dummy_task, dummy_task);
  ThreadPoolExecutor::SetNumberOfThreads(0);  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);
}

// Stop sleeping threads
TEST(ThreadPool, StopThreads_sleeping) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::SetNumberOfThreads(0);  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);
}

// Testing sync barriers *******************************************************

// Barrier waits until threads are done after parallel jobs
TEST_P(ParallelThreads, ImplicitBrriers) {
  uint64_t nthreads = GetParam();
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }

  int N_size = 100;
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  auto start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddParallelJobs(N_size, working_task);
  auto end = std::chrono::steady_clock::now();

  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, work_delay);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// After parallel recursive tasks
TEST_P(ParallelRecursion, ImplicitBrriers) {
  uint64_t depth = GetParam();
  uint64_t nthreads = (1ULL << (depth + 1)) - 2;
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);  // Implicit barrier
  auto start = std::chrono::steady_clock::now();
  recursive_calls(work_delay, depth, 0, 0);
  auto end = std::chrono::steady_clock::now();
  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, work_delay);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// One thread is sleeping
TEST(ThreadPool, ImplicitBrriers_1SleepingTask) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  auto start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddRecursiveCalls(
      0, 0,
      [](size_t s, size_t e) {
        HEXL_UNUSED(s);
        HEXL_UNUSED(e);
      },
      [](size_t s, size_t e) {
        HEXL_UNUSED(s);
        HEXL_UNUSED(e);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
      });
  auto end = std::chrono::steady_clock::now();
  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, 2 * HEXL_THREAD_WAIT_TIME);
  // Barrier work on sleeping threads

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Parallel Loops **************************************************************

// Test adding parallel loop jobs
TEST_P(ParallelThreads, ThreadIDs) {
  uint64_t nthreads = GetParam();
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }
  int N_size = 100;

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(0);
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  ThreadPoolExecutor::AddParallelJobs(N_size, id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add jobs on same thread pool when previous jobs are done
TEST(ThreadPool, AddParallelJob_after_done) {
  uint64_t nthreads = 2;
  int N_size = 100;
  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddParallelJobs(N_size, dummy_task);

  ThreadPoolExecutor::AddParallelJobs(N_size, id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add jobs when threads are sleeping
TEST(ThreadPool, AddParallelJob_after_sleeping) {
  uint64_t nthreads = 2;
  int N_size = 100;
  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::AddParallelJobs(N_size, id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Testing start and end parameters
TEST(ThreadPool, AddParallelJob_size_even) {
  uint64_t nthreads = 2;

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  std::list<int> expected = {0, 50, 50, 100};
  std::list<int> result;

  ThreadPoolExecutor::AddParallelJobs(100, [&](int start, int end) {
    std::lock_guard<std::mutex> lock(tasks_mutex);
    result.push_back(start);
    result.push_back(end);
  });
  result.sort();
  ASSERT_EQ(expected, result);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Testing start and end parameters
TEST(ThreadPool, AddParallelJob_size_odd) {
  uint64_t nthreads = 2;

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  std::list<int> expected = {0, 53, 53, 105};
  std::list<int> result;

  ThreadPoolExecutor::AddParallelJobs(105, [&](int start, int end) {
    std::lock_guard<std::mutex> lock(tasks_mutex);
    result.push_back(start);
    result.push_back(end);
  });
  result.sort();
  ASSERT_EQ(expected, result);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Testing start and end parameters
TEST(ThreadPool, AddParallelJob_size_small) {
  uint64_t nthreads = 2;

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  std::list<int> expected = {0, 1, 1, 1};
  std::list<int> result;

  ThreadPoolExecutor::AddParallelJobs(1, [&](int start, int end) {
    std::lock_guard<std::mutex> lock(tasks_mutex);
    result.push_back(start);
    result.push_back(end);
  });
  result.sort();
  ASSERT_EQ(expected, result);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Recursive Calls *************************************************************

// Test: Add nested tasks
TEST_P(ParallelRecursion, ThreadIDs) {
  uint64_t depth = GetParam();
  uint64_t nthreads = (1ULL << (depth + 1)) - 2;
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  recursive_calls(0, depth, 0, 0);

  task_ids.sort();
  ASSERT_EQ(task_ids.size(), nthreads + 1);  // calls
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), nthreads + 1);  // threads

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add tasks on same thread pool when previous jobs are done
// Using more than available threads
TEST(ThreadPool, AddRecursiveCalls_after_done) {
  uint64_t nthreads = 2;
  task_ids.clear();

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddRecursiveCalls(0, 0, dummy_task, dummy_task);

  ThreadPoolExecutor::AddRecursiveCalls(
      0, 0,
      [&](int id, int threads) {
        ThreadPoolExecutor::AddRecursiveCalls(1, 0, id_task, id_task);
        id_task(id, threads);
      },
      [&](int id, int threads) {
        ThreadPoolExecutor::AddRecursiveCalls(1, 1, id_task, id_task);
        id_task(id, threads);
      });

  task_ids.sort();
  ASSERT_EQ(task_ids.size(), 6);  // calls
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), nthreads);  // threads

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add tasks when threads are sleeping.
TEST(ThreadPool, AddRecursiveCalls_after_sleeping) {
  uint64_t nthreads = 2;
  task_ids.clear();

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::AddRecursiveCalls(0, 0, id_task, id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), nthreads);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test thread safety of the thread pool ***************************************

// Parallel Setup
TEST(ThreadPool, thread_safety_SetNumberOfThreads) {
  uint64_t nthreads = 2;
  sync.store(2);
  ThreadPoolExecutor::SetNumberOfThreads(0);

  std::thread thread_object1([nthreads]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::SetNumberOfThreads(nthreads >> 1);
  });
  std::thread thread_object2([nthreads]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  });
  thread_object1.join();
  thread_object2.join();
  uint64_t pool_size = ThreadPoolExecutor::GetNumberOfThreads();
  ASSERT_TRUE(pool_size == nthreads || pool_size == nthreads >> 1);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add nested tasks.
TEST_P(ParallelRecursion, Stress) {
  uint64_t depth = GetParam();
  uint64_t nthreads = (1ULL << (depth + 1)) - 2;
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  std::thread thread_object1([=]() {
    for (size_t i = 0; i < m_num_trials; i++) recursive_calls(1, depth, 0, 0);
  });
  std::thread thread_object2([=]() {
    for (size_t i = 0; i < m_num_trials; i++) recursive_calls(1, depth, 0, 0);
  });

  thread_object1.join();
  thread_object2.join();

  task_ids.sort();
  ASSERT_EQ(task_ids.size(), 2 * m_num_trials * (nthreads + 1));  // calls
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), nthreads + 2);  // threads

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add parallel jobs
TEST_P(ParallelThreads, Stress) {
  uint64_t nthreads = GetParam();
  if (nthreads > std::thread::hardware_concurrency()) {
    GTEST_SKIP();
  }
  int N_size = 100;
  iterations.store(0);
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  std::thread thread_object1([=]() {
    for (size_t i = 0; i < m_num_trials; i++)
      ThreadPoolExecutor::AddParallelJobs(N_size, add_iterations);
  });
  std::thread thread_object2([=]() {
    for (size_t i = 0; i < m_num_trials; i++)
      ThreadPoolExecutor::AddParallelJobs(N_size, add_iterations);
  });

  thread_object1.join();
  thread_object2.join();

  ASSERT_EQ(iterations.load(), 2 * m_num_trials * N_size);  // calls
  task_ids.unique();

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Parallel recursive task
TEST(ThreadPool, thread_safety_AddRecursiveCalls) {
  if (std::thread::hardware_concurrency() < 4) {
    GTEST_SKIP();
  }
  uint64_t nthreads = 4;
  sync.store(2);
  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  std::thread thread_object1([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddRecursiveCalls(0, 0, id_task, id_task);
  });
  std::thread thread_object2([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddRecursiveCalls(0, 0, id_task, id_task);
  });

  thread_object1.join();
  thread_object2.join();

  ASSERT_EQ(task_ids.size(), 4);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), 3);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Add task & stop threads in parallel
TEST(ThreadPool, thread_safety_AddJobs_n_stop) {
  HEXL_NUM_THREADS = 2;
  sync.store(2);
  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(0);

  std::thread thread_object1([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddRecursiveCalls(0, 0, id_task, id_task);
  });
  std::thread thread_object2([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::SetNumberOfThreads(0);
  });

  thread_object1.join();
  thread_object2.join();

  uint64_t pool_size = ThreadPoolExecutor::GetNumberOfThreads();
  ASSERT_TRUE(pool_size == 0 || pool_size == HEXL_NUM_THREADS);
  ASSERT_EQ(task_ids.size(), 2);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Parallel add jobs
TEST(ThreadPool, thread_safety_AddParallelJobs) {
  uint64_t nthreads = 2;
  sync.store(2);
  iterations.store(0);
  int N_size = 100;
  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  std::thread thread_object1([=]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddParallelJobs(N_size, id_task);
  });
  std::thread thread_object2([=]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddParallelJobs(N_size, id_task);
  });

  thread_object1.join();
  thread_object2.join();

  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), nthreads + 1);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Add jobs and setup threads in parallel
TEST(ThreadPool, thread_safety_AddJobs_n_setup) {
  uint64_t nthreads = 2;
  sync.store(2);
  iterations.store(0);
  int N_size = 100;
  ThreadPoolExecutor::SetNumberOfThreads(nthreads >> 1);

  std::thread thread_object1([=]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddParallelJobs(N_size, add_iterations);
  });
  std::thread thread_object2([nthreads]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  });

  thread_object1.join();
  thread_object2.join();

  uint64_t pool_size = ThreadPoolExecutor::GetNumberOfThreads();
  ASSERT_TRUE(pool_size == nthreads || pool_size == nthreads >> 1);
  ASSERT_EQ(iterations.load(), N_size);

  // *** Restore to some values in last test
  ThreadPoolExecutor::SetNumberOfThreads(2);
  HEXL_NTT_PARALLEL_DEPTH = 1;
}

// Test suites *****************************************************************
INSTANTIATE_TEST_SUITE_P(ThreadPool, ParallelRecursion,
                         ::testing::Values(0, 1, 2, 3, 4, 5));

INSTANTIATE_TEST_SUITE_P(ThreadPool, ParallelThreads,
                         ::testing::Values(0, 1, 2, 4, 8, 16, 32, 64));

#ifdef HEXL_DEBUG

// Testing debug features ******************************************************
TEST(ThreadPool, bad_input) {
  Task task([](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
  });

  EXPECT_ANY_THROW(ThreadPoolExecutor::AddParallelJobs(0, nullptr));

  EXPECT_ANY_THROW(ThreadPoolExecutor::AddRecursiveCalls(0, 0, nullptr, task));

  EXPECT_ANY_THROW(ThreadPoolExecutor::AddRecursiveCalls(0, 0, task, nullptr));
}
#endif  // HEXL_DEBUG

}  // namespace hexl
}  // namespace intel

#endif  // HEXL_MULTI_THREADING
