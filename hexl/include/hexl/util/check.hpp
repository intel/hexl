// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "hexl/util/types.hpp"

// Create logging/debug macros with no run-time overhead unless HEXL_DEBUG is
// enabled
#ifdef HEXL_DEBUG
#include "hexl/logging/logging.hpp"

#define HEXL_CHECK(cond, expr)                                       \
  if (!(cond)) {                                                     \
    LOG(ERROR) << expr << " in fuction: " << __FUNCTION__            \
               << " in file: " __FILE__ << " at line: " << __LINE__; \
    throw std::runtime_error("Error. Check log output");             \
  }

#define HEXL_CHECK_BOUNDS(arg, n, bound, expr)                            \
  for (size_t hexl_check_idx = 0; hexl_check_idx < n; ++hexl_check_idx) { \
    HEXL_CHECK((arg)[hexl_check_idx] < bound, expr);                      \
  }

#else  // HEXL_DEBUG=OFF

#define HEXL_CHECK(cond, expr) \
  {}
#define HEXL_CHECK_BOUNDS(...) \
  {}

#endif  // HEXL_DEBUG
