// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <omp.h>

#include <iostream>

#include "hexl/number-theory/number-theory.hpp"

namespace intel {
namespace hexl {

template <int InputModFactor>
void EltwiseFMAModNative(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                         const uint64_t* arg3, uint64_t n, uint64_t modulus) {
  uint64_t twice_modulus = 2 * modulus;
  uint64_t four_times_modulus = 4 * modulus;
  arg2 = ReduceMod<InputModFactor>(arg2, modulus, &twice_modulus,
                                   &four_times_modulus);

  MultiplyFactor mf(arg2, 64, modulus);


  if (arg3) {
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      uint64_t arg1_val = ReduceMod<InputModFactor>(
          arg1[i], modulus, &twice_modulus, &four_times_modulus);
      uint64_t arg3_val = ReduceMod<InputModFactor>(
          arg3[i], modulus, &twice_modulus, &four_times_modulus);

      uint64_t result_val =
          MultiplyMod(arg1_val, arg2, mf.BarrettFactor(), modulus);
      result[i] = AddUIntMod(result_val, arg3_val, modulus);
    }
  } else {  // arg3 == nullptr
#pragma omp parallel for
    for (size_t i = 0; i < n; ++i) {
      uint64_t arg1_val = ReduceMod<InputModFactor>(
          arg1[i], modulus, &twice_modulus, &four_times_modulus);
      result[i] = MultiplyMod(arg1_val, arg2, mf.BarrettFactor(), modulus);
    }
  }
  
}

}  // namespace hexl
}  // namespace intel
