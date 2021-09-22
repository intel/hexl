// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <random>
#include <utility>

#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/util.hpp"

namespace intel {
namespace hexl {

inline bool Compare(CMPINT cmp, uint64_t lhs, uint64_t rhs) {
  switch (cmp) {
    case CMPINT::EQ:
      return lhs == rhs;
    case CMPINT::LT:
      return lhs < rhs;
      break;
    case CMPINT::LE:
      return lhs <= rhs;
      break;
    case CMPINT::FALSE:
      return false;
      break;
    case CMPINT::NE:
      return lhs != rhs;
      break;
    case CMPINT::NLT:
      return lhs >= rhs;
      break;
    case CMPINT::NLE:
      return lhs > rhs;
    case CMPINT::TRUE:
      return true;
    default:
      return true;
  }
}

/// Generates a vector of size random values drawn uniformly from [min_value,
/// max_value)
/// NOTE: this function is not a cryptographically secure random number
/// generator and should be used for testing/benchmarking only
inline uint64_t GenerateInsecureUniformRandomValue(uint64_t min_value,
                                                   uint64_t max_value) {
  HEXL_CHECK(min_value < max_value, "min_value must be > max_value");

  static std::random_device rd;
  static std::mt19937 mersenne_engine(rd());
  std::uniform_int_distribution<uint64_t> distrib(min_value, max_value - 1);

  return distrib(mersenne_engine);
}

/// Generates a vector of size random values drawn uniformly from [min_value,
/// max_value)
/// NOTE: this function is not a cryptographically secure random
/// number generator and should be used for testing/benchmarking only
inline AlignedVector64<uint64_t> GenerateInsecureUniformRandomValues(
    uint64_t size, uint64_t min_value, uint64_t max_value) {
  AlignedVector64<uint64_t> values(size);
  auto generator = [&]() {
    return GenerateInsecureUniformRandomValue(min_value, max_value);
  };
  std::generate(values.begin(), values.end(), generator);
  return values;
}

}  // namespace hexl
}  // namespace intel
