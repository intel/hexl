// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-cmp-add.hpp"

#include <omp.h>

#include <iomanip>
#include <iostream>

#include "eltwise/eltwise-cmp-add-avx512.hpp"
#include "eltwise/eltwise-cmp-add-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"
namespace intel {
namespace hexl {

void EltwiseCmpAdd(uint64_t* result, const uint64_t* operand1, uint64_t n,
                   CMPINT cmp, uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(diff != 0, "Require diff != 0");

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    EltwiseCmpAddAVX512(result, operand1, n, cmp, bound, diff);
    return;
  }
#endif
  EltwiseCmpAddNative(result, operand1, n, cmp, bound, diff);
}

void EltwiseCmpAddNative(uint64_t* result, const uint64_t* operand1, uint64_t n,
                         CMPINT cmp, uint64_t bound, uint64_t diff) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(operand1 != nullptr, "Require operand1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0");
  HEXL_CHECK(diff != 0, "Require diff != 0");
  
  int thread_count;

  double start_time = omp_get_wtime();
  switch (cmp) {
    case CMPINT::EQ: {
#pragma omp parallel
    {
      thread_count = omp_get_num_threads();
      #pragma omp for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] == bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
    }
      break;
    }
    
    case CMPINT::LT:
#pragma omp parallel
    {
      thread_count = omp_get_num_threads();
      #pragma omp for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] < bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
    }
      break;

    case CMPINT::LE:
#pragma omp parallel
    {
      thread_count = omp_get_num_threads();
      #pragma omp for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] <= bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
  }
      break;

    case CMPINT::FALSE:
#pragma omp parallel
    {
      thread_count = omp_get_num_threads();
      #pragma omp for
      for (size_t i = 0; i < n; ++i) {
        result[i] = operand1[i];
      }
    }
      break;

    case CMPINT::NE:
#pragma omp parallel
    {
      thread_count = omp_get_num_threads();
      #pragma omp for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] != bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
    }
      break;

    case CMPINT::NLT:
#pragma omp parallel
    {
      thread_count = omp_get_num_threads();
      #pragma omp for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] >= bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
    }
      break;

    case CMPINT::NLE:
#pragma omp parallel
    {
      thread_count = omp_get_num_threads();
      #pragma omp for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] > bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
    }
      break;

    case CMPINT::TRUE:
#pragma omp parallel
    {
      thread_count = omp_get_num_threads();
      #pragma omp for
      for (size_t i = 0; i < n; ++i) {
        result[i] = operand1[i] + diff;
      }
    }
      break;

  }
  // Record the end time(timer2)
  double end_time = omp_get_wtime();

  // Calculate and print the elapsed time
  double elapsed_time = end_time - start_time;

  std::cout << thread_count << "  " << std::fixed << elapsed_time
            << std::setprecision(5) << std::endl;
}

}  // namespace hexl
}  // namespace intel
