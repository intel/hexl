// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <vector>

#include "hexl/util/defines.hpp"

// Wrap HEXL_VLOG with HEXL_DEBUG; this ensures no logging overhead in
// release mode
#ifdef HEXL_DEBUG

// TODO(fboemer) Enable if needed
// #define ELPP_THREAD_SAFE
#define ELPP_CUSTOM_COUT std::cerr
#define ELPP_STL_LOGGING
#define ELPP_LOG_STD_ARRAY
#define ELPP_LOG_UNORDERED_MAP
#define ELPP_LOG_UNORDERED_SET
#define ELPP_NO_LOG_TO_FILE
#define ELPP_DISABLE_DEFAULT_CRASH_HANDLING
#define ELPP_WINSOCK2

#include <easylogging++.h>

#define HEXL_VLOG(N, rest) \
  do {                     \
    if (VLOG_IS_ON(N)) {   \
      VLOG(N) << rest;     \
    }                      \
  } while (0);

#else

#define HEXL_VLOG(N, rest) \
  {}

#define START_EASYLOGGINGPP(X, Y) \
  {}

#endif
