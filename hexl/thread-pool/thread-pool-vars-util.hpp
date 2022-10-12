// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <charconv>
#include <cstdlib>
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
static int env_var_to_int(const char* var) {
  int value = 0;

  // Get value from env variable
  char* var_value = std::getenv(var);
  if (var_value != nullptr) {
    std::string str = var_value;
    std::from_chars(str.data(), str.data() + str.size(), value);
  }
  return value;
}

// Verify for appropriate number of threads
static int setup_num_threads(const char* var) {
  int hw_val = std::thread::hardware_concurrency();
  int value = env_var_to_int(var);

  // Use default value in case of error
  if (value <= 0) {
    value = HEXL_DEFAULT_NUM_THREADS;
  }

  // Check max threads available
  if (value > hw_val) {
    value = hw_val;
  }
  return value;
}

// Verify for appropriate number of recursive calls
static int setup_ntt_calls(const char* var) {
  int value = env_var_to_int(var);
  uint64_t threads;

  // Use default value in case of error
  if (value <= 0) {
    value = HEXL_DEFAULT_NTT_PARALLEL_DEPTH;
  }

  // Sum of powers of 2 minus main thread
  threads = (1ULL << (value + 1)) - 2;

  // Check max threads available
  if (threads > HEXL_NUM_THREADS) {
    value = 0;
  }

  return value;
}
#endif
}  // namespace hexl
}  // namespace intel
