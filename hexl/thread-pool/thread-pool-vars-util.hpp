// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <math.h>

#include <cstdlib>
#include <string>
#include <thread>

#include "hexl/logging/logging.hpp"
#include "hexl/thread-pool/thread-pool-vars.hpp"

namespace intel {
namespace hexl {

const uint HEXL_DEFAULT_NUM_THREADS = 16;
const uint HEXL_DEFAULT_NTT_PARALLEL_DEPTH = 1;

#ifdef HEXL_MULTI_THREADING

// Check for environment variable
static int check_env_var(const char* var) {
  int value = 0;

  // Get value from env variable
  char* var_value = std::getenv(var);
  if (var_value != NULL) {
    value = static_cast<int>(std::strtol(var_value, NULL, 10));
  }
  return value;
}

// Verify for appropriate number of threads
static int setup_num_threads(const char* var) {
  int hw_val = std::thread::hardware_concurrency();
  int value = check_env_var(var);

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
  int value = check_env_var(var);
  uint threads;

  // Use default value in case of error
  if (value <= 0) {
    value = HEXL_DEFAULT_NTT_PARALLEL_DEPTH;
  }

  // Sum of powers of 2 minus main thread
  threads = static_cast<int>(pow(2, value + 1)) - 2;

  // Check max threads available
  if (threads > HEXL_NUM_THREADS) {
    value = 0;
  }

  return value;
}
#endif
}  // namespace hexl
}  // namespace intel
