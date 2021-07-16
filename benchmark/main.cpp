// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <benchmark/benchmark.h>

#include "hexl/logging/logging.hpp"

namespace intel {
namespace hexl {

void register_eltwise_add_mod_benchmarks();
void register_eltwise_cmp_add_benchmarks();
void register_eltwise_cmp_sub_mod_benchmarks();
void register_eltwise_fma_mod_benchmarks();
void register_eltwise_mult_mod_benchmarks();
void register_eltwise_sub_mod_benchmarks();
void register_eltwise_reduce_mod_benchmarks();
void register_ntt_benchmarks();

}  // namespace hexl
}  // namespace intel

int main(int argc, char** argv) {
  START_EASYLOGGINGPP(argc, argv);

  intel::hexl::register_eltwise_add_mod_benchmarks();
  intel::hexl::register_eltwise_fma_mod_benchmarks();
  intel::hexl::register_eltwise_cmp_add_benchmarks();
  intel::hexl::register_eltwise_cmp_sub_mod_benchmarks();
  intel::hexl::register_eltwise_mult_mod_benchmarks();
  intel::hexl::register_eltwise_sub_mod_benchmarks();
  intel::hexl::register_eltwise_reduce_mod_benchmarks();
  intel::hexl::register_ntt_benchmarks();

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();

  return 0;
}
