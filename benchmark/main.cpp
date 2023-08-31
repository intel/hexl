// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>
#include <omp.h>
#include <iostream>
#include "hexl/logging/logging.hpp"

int main(int argc, char** argv) {
  int max_threads = omp_get_max_threads();
  std::cout << "Maximum number of threads = " << max_threads << std::endl;

  START_EASYLOGGINGPP(argc, argv);

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
