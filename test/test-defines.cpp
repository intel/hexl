// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "hexl/util/defines.hpp"

TEST(HEXL_REPEAT, 8) {
  int x = 0;
  HEXL_REPEAT(x++, 8);
  ASSERT_EQ(x, 8);
}
