// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "util/types.hpp"

// Create logging/debug macros with no run-time overhead unless HEXL_DEBUG is
// enabled
#ifdef HEXL_DEBUG
#include "logging/logging.hpp"

#define HEXL_CHECK(cond, expr)                                       \
  if (!(cond)) {                                                     \
    LOG(ERROR) << expr << " in fuction: " << __FUNCTION__            \
               << " in file: " __FILE__ << " at line: " << __LINE__; \
    throw std::runtime_error("Error. Check log output");             \
  }

#define HEXL_CHECK_BOUNDS(arg, n, bound, expr) \
  for (size_t i = 0; i < n; ++i) {             \
    HEXL_CHECK((arg)[i] < bound, expr);        \
  }

#else  // HEXL_DEBUG=OFF

#define HEXL_CHECK(cond, expr) \
  {}
#define HEXL_CHECK_BOUNDS(...) \
  {}

#endif  // HEXL_DEBUG
