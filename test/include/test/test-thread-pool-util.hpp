// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <gtest/gtest.h>

#include "hexl/util/check.hpp"
#include "thread-pool/thread-pool-executor.hpp"

namespace intel {
namespace hexl {

// Parameters = (recursive depth)
class ParallelRecursion : public ::testing::TestWithParam<uint64_t> {
 protected:
  void SetUp() {
#ifdef HEXL_DEBUG
    m_num_trials = 1;
#else
    m_num_trials = 1000;
#endif
  }
  void TearDown() {}

 public:
  uint64_t m_num_trials;
};

// Parameters = (number of threads)
class ParallelThreads : public ::testing::TestWithParam<uint64_t> {
 protected:
  void SetUp() {
#ifdef HEXL_DEBUG
    m_num_trials = 1;
#else
    m_num_trials = 1000;
#endif
  }
  void TearDown() {}

 public:
  uint64_t m_num_trials;
};

}  // namespace hexl
}  // namespace intel
