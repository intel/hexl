// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include "hexl/logging/logging.hpp"

int main(int argc, char** argv) {
  START_EASYLOGGINGPP(argc, argv);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
