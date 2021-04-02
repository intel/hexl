// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

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

}  // namespace hexl
}  // namespace intel
