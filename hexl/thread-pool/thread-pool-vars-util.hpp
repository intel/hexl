// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "hexl/logging/logging.hpp"
#include "hexl/thread-pool/thread-pool-vars.hpp"

#pragma once

namespace intel {
namespace hexl {

#ifdef HEXL_MULTI_THREADING
static int check_env_var(const char* var) {
  int value = -1;

  // Get value from env variable
  char* var_value = std::getenv(var);
  if (var_value != NULL) {
    value = static_cast<int>(std::strtol(var_value, NULL, 10));
  }
  return value;
}

static int setup_num_threads(const char* var) {
  int hw_val = std::thread::hardware_concurrency();
  int value = check_env_var(var);

  // Use default value in case of error
  if (value < 0) {
    value = 16;
    HEXL_VLOG(3, "Using default number of threads.");
  }
  // Check max threads available
  if (value > hw_val) {
    value = hw_val;
    HEXL_VLOG(3, "Threads reduced to platform's maximum number of threads.");
  }

  HEXL_VLOG(2, "Using " << value << " threads for thread pool.");
  return value;
}

static int setup_ntt_calls(const char* var) {
  int value = check_env_var(var);
  int threads;

  // Use default value in case of error
  if (value < 0) {
    value = 1;
    HEXL_VLOG(3, "Using default NTT's parallel depth.");
  }

  threads = static_cast<int>(pow(2, value + 1)) -
            2;  // Sum of powers of 2 minus main thread

  // Check max threads available
  if (threads > HEXL_NUM_THREADS) {
    HEXL_VLOG(3, "Provided NTT's parallel depth exceeds number of threads.");
    value = 0;
  }

  HEXL_VLOG(2, "Using depth " << value << " for parallel NTT calls.");
  return value;
}
#endif
}  // namespace hexl
}  // namespace intel
