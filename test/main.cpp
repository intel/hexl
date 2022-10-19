// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "hexl/logging/logging.hpp"
#include "thread-pool/thread-pool-executor.hpp"

int main(int argc, char** argv) {
  START_EASYLOGGINGPP(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  intel::hexl::ThreadPoolExecutor::SetNumberOfThreads(2);
  int rc = RUN_ALL_TESTS();
  return rc;
}
