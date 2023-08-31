// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-cmp-add.hpp"

#include <omp.h>

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

  switch (cmp) {
    case CMPINT::EQ: {
    #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] == bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;
    }
    
    case CMPINT::LT:
    #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] < bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;

    case CMPINT::LE:
    #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] <= bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;

    case CMPINT::FALSE:
    #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        result[i] = operand1[i];
      }
      break;

    case CMPINT::NE:
    #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] != bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;

    case CMPINT::NLT:
    #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] >= bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;

    case CMPINT::NLE:
    #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        if (operand1[i] > bound) {
          result[i] = operand1[i] + diff;
        } else {
          result[i] = operand1[i];
        }
      }
      break;

    case CMPINT::TRUE:
    #pragma omp parallel for
      for (size_t i = 0; i < n; ++i) {
        result[i] = operand1[i] + diff;
      }
      break;
  }
  
}

}  // namespace hexl
}  // namespace intel
