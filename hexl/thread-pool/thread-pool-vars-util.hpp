// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "hexl/logging/logging.hpp"
#include "hexl/thread-pool/thread-pool-vars.hpp"

namespace intel {
namespace hexl {

constexpr uint64_t HEXL_DEFAULT_NUM_THREADS = 16;
constexpr uint64_t HEXL_DEFAULT_NTT_PARALLEL_DEPTH = 1;

#ifdef HEXL_MULTI_THREADING

// Check for environment variable
static int64_t env_var_to_int(const char* var) {
  // Get value from env variable
  char* var_value = std::getenv(var);
  if (var_value == nullptr) {
    return 0;  // returns 0 if unset
  }

  errno = 0;
  int64_t value = std::strtol(var_value, nullptr, 10);
  std::cout << "value " << value << std::endl;

  // Checks
  if (errno == ERANGE) {
    std::cout << "ERROR: Env variable '" << var << "=" << var_value;
    std::cout << "' is out of range." << std::endl;
    exit(1);
  }

  if (value <= 0) {
    std::cout << "ERROR: Env variable '" << var << "=" << var_value;
    std::cout << "' is not valid." << std::endl;
    exit(1);
  }

  return (value < 0L) ? 0L : value;  // not negative number
}

// Verify for appropriate number of threads
static int64_t setup_num_threads(const char* var) {
  int64_t value = env_var_to_int(var);

  // Use default value in case of error
  if (value == 0) {
    value = HEXL_DEFAULT_NUM_THREADS;
  }

  // Check max threads available
  if (int64_t hw_val = std::thread::hardware_concurrency(); value > hw_val) {
    value = hw_val;
  }
  return value;
}

// Verify for appropriate number of recursive calls
static int64_t setup_ntt_calls(const char* var) {
  int64_t value = env_var_to_int(var);

  // Use default value in case of error
  if (value <= 0) {
    value = HEXL_DEFAULT_NTT_PARALLEL_DEPTH;
  }

  // Sum of powers of 2 minus main thread
  uint64_t threads = (1ULL << (value + 1)) - 2;

  // Check max threads available
  if (threads > HEXL_NUM_THREADS) {
    value = 0;
  }

  return value;
}
#endif
}  // namespace hexl
}  // namespace intel
