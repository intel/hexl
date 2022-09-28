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
  uint threads = 4;
  ThreadPoolExecutor::StopThreads();
  ThreadPoolExecutor::SetNumberOfThreads(threads);

  // When setup. Correspond to SetNumberOfThreads.
  auto handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), threads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  // When running parallel jobs. Keep the same value.
  ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  });
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), threads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  // When running parallel task. Keep the same value.
  ThreadPoolExecutor::AddTask([](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  });
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), threads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  // When all done. Keep the same value.
  ThreadPoolExecutor::SetBarrier();
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), threads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  // When sleeping. Keep the same value.
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), threads);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  // When stopped. Returns zero.
  ThreadPoolExecutor::StopThreads();
  handlers = ThreadPoolExecutor::GetThreadHandlers();
  ASSERT_EQ(handlers.size(), 0);
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), handlers.size());

  ThreadPoolExecutor::StopThreads();
}

// Testing function that sets number of threads from env variable
TEST(ThreadPool, setup_num_threads_env_var) {
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

  // Undefined: Default value is set
  {
    char env[] = "HEXL_NUM_THREADS";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, HEXL_DEFAULT_NUM_THREADS);
  }

  // Negative: Default value is set
  {
    char env[] = "HEXL_NUM_THREADS=-2";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, HEXL_DEFAULT_NUM_THREADS);
  }

  // Floating point: Rounded value is set
  {
    char env[] = "HEXL_NUM_THREADS=4.5";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, 4);
  }

  // String: Default value is set
  {
    char env[] = "HEXL_NUM_THREADS=error";
    putenv(env);
    auto value = setup_num_threads("HEXL_NUM_THREADS");
    ASSERT_EQ(value, HEXL_DEFAULT_NUM_THREADS);
  }

  // Env var is used for thread pool size
  // Set to different than default or previously set value
  {
    char env[] = "HEXL_NUM_THREADS=8";
    putenv(env);
    HEXL_NUM_THREADS = setup_num_threads("HEXL_NUM_THREADS");
    ThreadPoolExecutor::StopThreads();
    ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
    });
    auto value = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(value, 8);
  }

  ThreadPoolExecutor::StopThreads();
}

// Testing function that sets number of parallel ntt calls from env variable
TEST(ThreadPool, setup_ntt_calls_env_var) {
  // Overshooting HEXL_NTT_PARALLEL_DEPTH: Zero is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH=999999999";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, 0);
  }

  // Wanted value is set
  {
    char env[] = "HEXL_NTT_PARALLEL_DEPTH=2";
    putenv(env);
    auto value = setup_ntt_calls("HEXL_NTT_PARALLEL_DEPTH");
    ASSERT_EQ(value, 2);
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
    ThreadPoolExecutor::StopThreads();
    ThreadPoolExecutor::SetNumberOfThreads(999999999);
    auto value = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(value, std::thread::hardware_concurrency());
  }

  {
    // Presedence over env variable
    uint threads = 4;
    HEXL_NUM_THREADS = 2;
    ThreadPoolExecutor::StopThreads();
    ThreadPoolExecutor::SetNumberOfThreads(threads);
    auto value = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(value, threads);

    // Setting new bigger value
    threads = 6;
    ThreadPoolExecutor::SetNumberOfThreads(threads);
    value = ThreadPoolExecutor::GetNumberOfThreads();
    ASSERT_EQ(value, threads);

    // Setting new smaller value. TODO
  }

  {
    // Test: N threads get started
    int counter = 0;
    ThreadPoolExecutor::StopThreads();
    ThreadPoolExecutor::SetNumberOfThreads(4);
    auto handlers = ThreadPoolExecutor::GetThreadHandlers();
    for (size_t i = 0; i < handlers.size(); i++) {
      auto handler = handlers.at(i);
      if (handler->state.load() == STATE::DONE) {
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

  ThreadPoolExecutor::StopThreads();
}

// Test synchornization barrier
TEST(ThreadPool, SetBrrier) {
  int delay = 2;
  ThreadPoolExecutor::StopThreads();
  ThreadPoolExecutor::SetNumberOfThreads(4);  // Implicit barrier

  // Barrier waits 'til threads are done after parallel jobs
  auto start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::AddParallelJobs([delay](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  });  // Implicit Barrier
  auto end = std::chrono::steady_clock::now();

  uint64_t duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  ASSERT_GE(duration, delay);

  // After parallel tasks with explicit barrier
  start = std::chrono::steady_clock::now();
  for (uint i = 0; i < ThreadPoolExecutor::GetNumberOfThreads(); i++) {
    ThreadPoolExecutor::AddTask([delay](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::this_thread::sleep_for(std::chrono::milliseconds(delay));
    });
  }
  ThreadPoolExecutor::SetBarrier();  // Waits 'till jobs are done
  end = std::chrono::steady_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count();
  ASSERT_NEAR(duration, delay + 1, 1);

  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));

  // Quickly pass barrier if on sleep mode. No wait.
  start = std::chrono::steady_clock::now();
  ThreadPoolExecutor::SetBarrier();
  end = std::chrono::steady_clock::now();

  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count();
  ASSERT_EQ(duration, 0);
  // Barrier work on sleeping threads

  ThreadPoolExecutor::StopThreads();
}

// Test threads are deleted
TEST(ThreadPool, StopThreads) {
  ThreadPoolExecutor::StopThreads();

  // Stop waiting threads
  ThreadPoolExecutor::SetNumberOfThreads(4);
  ThreadPoolExecutor::StopThreads();
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);

  // Stop working threads
  ThreadPoolExecutor::SetNumberOfThreads(4);
  ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  });
  ThreadPoolExecutor::StopThreads();  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);

  ThreadPoolExecutor::SetNumberOfThreads(4);
  for (uint i = 0; i < ThreadPoolExecutor::GetNumberOfThreads(); i++) {
    ThreadPoolExecutor::AddTask([](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
  }
  ThreadPoolExecutor::StopThreads();  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);

  // Stop sleeping threads
  ThreadPoolExecutor::SetNumberOfThreads(4);
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::StopThreads();  // Stop when jobs finish
  ASSERT_EQ(ThreadPoolExecutor::GetNumberOfThreads(), 0);
}

// Test adding parallel loop jobs
TEST(ThreadPool, AddParallelJob) {
  std::list<std::thread::id> ids;
  std::mutex list_mutex;
  HEXL_NUM_THREADS = 4;

  // Add jobs without previous setup
  ThreadPoolExecutor::AddParallelJobs([&list_mutex, &ids](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::lock_guard<std::mutex> lock(list_mutex);
    ids.push_back(std::this_thread::get_id());
  });
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ids.clear();
  ThreadPoolExecutor::StopThreads();

  // Add jobs with previous setup different than env variable
  ThreadPoolExecutor::SetNumberOfThreads(8);
  ThreadPoolExecutor::AddParallelJobs([&list_mutex, &ids](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::lock_guard<std::mutex> lock(list_mutex);
    ids.push_back(std::this_thread::get_id());
  });
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ids.clear();

  // Add jobs on same thread pool when previous jobs are done
  ThreadPoolExecutor::AddParallelJobs([&list_mutex, &ids](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::lock_guard<std::mutex> lock(list_mutex);
    ids.push_back(std::this_thread::get_id());
  });
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ids.clear();

  // Add jobs when threads are sleeping
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  ThreadPoolExecutor::AddParallelJobs([&list_mutex, &ids](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::lock_guard<std::mutex> lock(list_mutex);
    ids.push_back(std::this_thread::get_id());
  });
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), ThreadPoolExecutor::GetNumberOfThreads());

  ids.clear();

  // Add jobs when threads are busy
  std::thread thread_object([]() {
    ThreadPoolExecutor::AddParallelJobs([](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    });
  });

  // Give time for previous threads to be running
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  ThreadPoolExecutor::AddParallelJobs([&list_mutex, &ids](int id, int threads) {
    HEXL_UNUSED(id);
    HEXL_UNUSED(threads);
    std::lock_guard<std::mutex> lock(list_mutex);
    ids.push_back(std::this_thread::get_id());
  });

  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), 1);

  // Testing id and threads parameters
  std::list<int> expected, result;
  ThreadPoolExecutor::SetBarrier();  // Needed to wait on nested threads
  int threads = ThreadPoolExecutor::GetNumberOfThreads();
  for (int i = 0; i < threads; i++) expected.push_back(i);
  for (int i = 0; i < threads; i++) expected.push_back(threads);
  ThreadPoolExecutor::AddParallelJobs(
      [&list_mutex, &result](int id, int threads) {
        std::lock_guard<std::mutex> lock(list_mutex);
        result.push_back(id);
        result.push_back(threads);
      });
  result.sort();
  ASSERT_EQ(expected, result);

  ThreadPoolExecutor::StopThreads();

  thread_object.join();
}

// Test adding parallel tasks
TEST(ThreadPool, AddTask) {
  std::list<std::thread::id> ids;
  std::mutex list_mutex;
  HEXL_NUM_THREADS = 4;

  // Test: Add tasks without previous setup. Less than available threads
  int threads = HEXL_NUM_THREADS - 1;
  for (int i = 0; i < threads; i++) {
    ThreadPoolExecutor::AddTask([&list_mutex, &ids](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::lock_guard<std::mutex> lock(list_mutex);
      ids.push_back(std::this_thread::get_id());
    });
  }
  ThreadPoolExecutor::SetBarrier();
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), threads);

  ids.clear();
  ThreadPoolExecutor::StopThreads();

  // Test: Add tasks with previous setup different than env var.
  // Using all threads.
  threads = 8;
  ThreadPoolExecutor::SetNumberOfThreads(threads);
  for (int i = 0; i < threads; i++) {
    ThreadPoolExecutor::AddTask([&list_mutex, &ids](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::lock_guard<std::mutex> lock(list_mutex);
      ids.push_back(std::this_thread::get_id());
    });
  }
  ThreadPoolExecutor::SetBarrier();
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), threads);

  ids.clear();

  // Test: Add tasks on same thread pool when previous jobs are done.
  // Ensured by barrier on previous thread.
  // Using more than available threads
  for (int i = 0; i < threads + 3; i++) {
    ThreadPoolExecutor::AddTask([&list_mutex, &ids](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::lock_guard<std::mutex> lock(list_mutex);
      ids.push_back(std::this_thread::get_id());
    });
  }
  ThreadPoolExecutor::SetBarrier();
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), threads + 1);  // threads + main thread

  ids.clear();

  // Test: Add tasks when threads are sleeping. Threads from previous test.
  // Wait for threads to sleep
  std::this_thread::sleep_for(
      std::chrono::milliseconds(2 * HEXL_THREAD_WAIT_TIME));
  for (int i = 0; i < threads; i++) {
    ThreadPoolExecutor::AddTask([&list_mutex, &ids](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::lock_guard<std::mutex> lock(list_mutex);
      ids.push_back(std::this_thread::get_id());
    });
  }
  ThreadPoolExecutor::SetBarrier();
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), threads);

  ids.clear();

  // Test: Add tasks when threads are busy
  // Making threads busy
  for (int i = 0; i < threads; i++) {
    ThreadPoolExecutor::AddTask([](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    });
  }
  // Adding tasks
  for (int i = 0; i < threads; i++) {
    ThreadPoolExecutor::AddTask([&list_mutex, &ids](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::lock_guard<std::mutex> lock(list_mutex);
      ids.push_back(std::this_thread::get_id());
    });
  }
  ThreadPoolExecutor::SetBarrier();
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), 1);

  ids.clear();

  // Test: Add tasks when threads are done without preceding barrier
  // Using threads once
  for (int i = 0; i < threads; i++) {
    ThreadPoolExecutor::AddTask([](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
    });
  }
  // No Barrier
  // Wait for threads just to be done
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  // Trying to use threads again
  for (int i = 0; i < threads; i++) {
    ThreadPoolExecutor::AddTask([&list_mutex, &ids](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      std::lock_guard<std::mutex> lock(list_mutex);
      ids.push_back(std::this_thread::get_id());
    });
  }
  ThreadPoolExecutor::SetBarrier();
  ids.sort();
  ids.unique();
  ASSERT_EQ(ids.size(), 1);

  // Testing id and threads parameters
  std::list<int> expected, result;
  threads = ThreadPoolExecutor::GetNumberOfThreads();
  for (int i = 0; i < threads; i++) expected.push_back(i);
  for (int i = 0; i < threads; i++) expected.push_back(threads);
  for (int i = 0; i < threads; i++) {
    ThreadPoolExecutor::AddTask([&list_mutex, &result](int id, int threads) {
      HEXL_UNUSED(id);
      HEXL_UNUSED(threads);
      {
        std::lock_guard<std::mutex> lock(list_mutex);
        result.push_back(id);
        result.push_back(threads);
      }
    });
  }
  ThreadPoolExecutor::SetBarrier();
  result.sort();
  ASSERT_EQ(expected, result);

  ThreadPoolExecutor::StopThreads();
}

#ifdef HEXL_DEBUG
// Testing debug features
TEST(ThreadPool, bad_input) {
  // EXPECT_ANY_THROW(ThreadPoolExecutor::SetNumberOfThreads(2););
}
#endif

#endif
}  // namespace hexl
}  // namespace intel
