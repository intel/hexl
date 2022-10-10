// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "test/test-thread-pool-util.hpp"
#include "test/test-util.hpp"
#include "thread-pool/thread-pool-executor.hpp"
#include "thread-pool/thread-pool-vars-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_MULTI_THREADING

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

// Testing number of threads across different phases

// After setup. Correspond to SetNumberOfThreads.
TEST(ThreadPool, GetNumberOfThreads_after_setup) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(0);

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// After stopped. Returns zero.
TEST(ThreadPool, GetNumberOfThreads_after_stop) {
  uint64_t nthreads = 2;
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  ThreadPoolExecutor::SetNumberOfThreads(0);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), 0);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());
}

// After running parallel jobs. Without previous setup.
TEST(ThreadPool, GetNumberOfThreads_after_AddParallelJobs) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(0);

  HEXL_NUM_THREADS = nthreads;
  ThreadPoolExecutor::AddParallelJobs(dummy_task);
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
  ThreadPoolExecutor::AddRecursiveCalls(dummy_task, dummy_task);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// After sleeping. Keep the same value.
TEST(ThreadPool, GetNumberOfThreads_after_sleep) {
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

// Using 50% of available threads
TEST(ThreadPool, GetNumberOfThreads_50p_threads) {
  uint64_t nthreads =
      static_cast<uint64_t>(std::thread::hardware_concurrency() * 0.5);

  // Try at 50% of available threads
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), nthreads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test setting number of threads programmatically

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
TEST(ThreadPool, SetNumberOfThreads_set) {
  uint64_t nthreads = 2;
  uint64_t counter = 0;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  for (size_t i = 0; i < handlers.size(); i++) {
    auto handler = handlers.at(i);
    if (handler->state.load() == static_cast<int>(STATE::DONE) ||
        handler->state.load() == static_cast<int>(STATE::SLEEPING)) {
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
TEST(ThreadPool, SetNumberOfThreads_sleeping) {
  uint64_t nthreads = 2;
  uint64_t counter = 0;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  for (size_t i = 0; i < handlers.size(); i++) {
    auto handler = handlers.at(i);
    if (handler->state.load() == static_cast<int>(STATE::SLEEPING)) {
      counter++;
    }
  }
  ASSERT_EQ(counter, ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test threads are deleted
// Stop ready threads
TEST(ThreadPool, StopThreads_ready) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::SetNumberOfThreads(0);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);
}

// Stop done threads after parallel jobs
TEST(ThreadPool, StopThreads_after_AddParallelJobs) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  });
  ThreadPoolExecutor::SetNumberOfThreads(0);  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);
}

// Stop done threads after recursive tasks
TEST(ThreadPool, StopThreads_after_AddRecursiveCalls) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddRecursiveCalls(dummy_task, dummy_task);
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

// Test synchronization barrier
TEST(ThreadPool, ImplicitBrriers_setup) {
  uint64_t nthreads = 2;
  uint64_t counter = 0;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);  // Implicit barrier

  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  for (size_t i = 0; i < handlers.size(); i++) {
    auto handler = handlers.at(i);
    if (handler->state.load() == static_cast<int>(STATE::DONE) ||
        handler->state.load() == static_cast<int>(STATE::SLEEPING)) {
      counter++;
    }
  }
  ASSERT_EQ(counter, ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Barrier waits 'til threads are done after parallel jobs
TEST(ThreadPool, ImplicitBrriers_AddParallelJobs) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  auto start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddParallelJobs(working_task);
  auto end = std::chrono::steady_clock::now();

  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, work_delay);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// After parallel recursive tasks
TEST(ThreadPool, ImplicitBrriers_AddRecursiveCalls) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  auto start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddRecursiveCalls(working_task, working_task);
  auto end = std::chrono::steady_clock::now();

  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, work_delay);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// One thread is sleeping
TEST(ThreadPool, ImplicitBrriers_Sleeping) {
  uint64_t nthreads = 2;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  auto start = std::chrono::steady_clock::now();
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
  auto end = std::chrono::steady_clock::now();
  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, 2 * HEXL_THREAD_WAIT_TIME);
  // Barrier work on sleeping threads

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// On nested tasks
TEST(ThreadPool, ImplicitBrriers_NestedTasks) {
  uint64_t nthreads = 6;

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);  // Implicit barrier
  auto start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddRecursiveCalls(
      [=](size_t id, size_t threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        ThreadPoolExecutor::AddRecursiveCalls(working_task, working_task);
      },
      [=](size_t id, size_t threads) {
        HEXL_UNUSED(id);
        HEXL_UNUSED(threads);
        ThreadPoolExecutor::AddRecursiveCalls(working_task, working_task);
      });
  auto end = std::chrono::steady_clock::now();
  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, work_delay);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test adding parallel loop jobs
TEST(ThreadPool, AddParallelJob_threads_new) {
  uint64_t nthreads = 2;

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(0);
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  ThreadPoolExecutor::AddParallelJobs(id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add jobs on same thread pool when previous jobs are done
TEST(ThreadPool, AddParallelJob_threads_done) {
  uint64_t nthreads = 2;

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddParallelJobs(dummy_task);

  ThreadPoolExecutor::AddParallelJobs(id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add jobs when threads are sleeping
TEST(ThreadPool, AddParallelJob_threads_sleeping) {
  uint64_t nthreads = 2;

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::AddParallelJobs(id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Testing id and threads parameters
TEST(ThreadPool, AddParallelJob_threads_ids) {
  uint64_t nthreads = 2;

  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  std::list<int> expected, result;
  for (size_t i = 0; i < nthreads; i++) {
    expected.push_back(i);
    expected.push_back(nthreads);
  }
  ThreadPoolExecutor::AddParallelJobs([&](int id, int threads) {
    std::lock_guard<std::mutex> lock(tasks_mutex);
    result.push_back(id);
    result.push_back(threads);
  });
  expected.sort();
  result.sort();
  ASSERT_EQ(expected, result);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: 50% of available threads
TEST(ThreadPool, AddParallelJob_50p_threads) {
  uint64_t nthreads =
      static_cast<uint64_t>(std::thread::hardware_concurrency() * 0.5);

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  ThreadPoolExecutor::AddParallelJobs(id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test adding parallel tasks
TEST(ThreadPool, AddRecursiveCalls_threads_new) {
  uint64_t nthreads = 2;
  task_ids.clear();

  ThreadPoolExecutor::SetNumberOfThreads(0);
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);  // Setup
  ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), nthreads);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add tasks on same thread pool when previous jobs are done
// Using more than available threads
TEST(ThreadPool, AddRecursiveCalls_threads_done) {
  uint64_t nthreads = 2;
  task_ids.clear();

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  ThreadPoolExecutor::AddRecursiveCalls(
      [&](int id, int threads) {
        ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
        id_task(id, threads);
      },
      [&](int id, int threads) {
        ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
        id_task(id, threads);
      });

  task_ids.sort();
  ASSERT_EQ(task_ids.size(), 6);  // calls
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), nthreads);  // threads

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test: Add tasks when threads are sleeping.
TEST(ThreadPool, AddRecursiveCalls_threads_sleeping) {
  uint64_t nthreads = 2;
  task_ids.clear();

  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
  task_ids.sort();
  task_ids.unique();
  ASSERT_EQ(task_ids.size(), nthreads);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Test thread safety of the thread pool
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

// Parallel recursive task
TEST(ThreadPool, thread_safety_AddRecursiveCalls) {
  uint64_t nthreads = 4;
  sync.store(2);
  task_ids.clear();
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);

  std::thread thread_object1([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
  });
  std::thread thread_object2([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
  });

  thread_object1.join();
  thread_object2.join();

  int counter = 0;
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  for (size_t i = 0; i < handlers.size(); i++) {
    auto handler = handlers.at(i);
    if (handler->state.load() == static_cast<int>(STATE::DONE)) {
      counter++;
    }
  }
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
    ThreadPoolExecutor::AddRecursiveCalls(id_task, id_task);
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
  ThreadPoolExecutor::SetNumberOfThreads(nthreads);
  std::thread thread_object1([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddParallelJobs(add_iterations);
  });
  std::thread thread_object2([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddParallelJobs(add_iterations);
  });

  thread_object1.join();
  thread_object2.join();

  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), nthreads);
  ASSERT_EQ(iterations.load(), 2 * N_size);

  ThreadPoolExecutor::SetNumberOfThreads(0);
}

// Add jobs and setup threads in parallel
TEST(ThreadPool, thread_safety_AddJobs_n_setup) {
  uint64_t nthreads = 2;
  sync.store(2);
  iterations.store(0);
  ThreadPoolExecutor::SetNumberOfThreads(nthreads >> 1);

  std::thread thread_object1([]() {
    sync.fetch_add(-1);
    while (sync) {
    }
    ThreadPoolExecutor::AddParallelJobs(add_iterations);
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

  ThreadPoolExecutor::SetNumberOfThreads(2);
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
