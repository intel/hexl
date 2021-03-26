// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"
#include "logging/logging.hpp"
#include "test-util.hpp"
#include "util/aligned-allocator.hpp"
#include "util/types.hpp"

namespace intel {
namespace hexl {

TEST(AlignedVector64, alloc) {
  AlignedVector64<uint64_t> x{1, 2, 3, 4};
  ASSERT_EQ(reinterpret_cast<uintptr_t>(x.data()) % 64, 0);
}

TEST(AlignedVector64, assignment) {
  AlignedVector64<uint64_t> x{1, 2, 3, 4};
  AlignedVector64<uint64_t> y = x;
  ASSERT_EQ(reinterpret_cast<uintptr_t>(x.data()) % 64, 0);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(y.data()) % 64, 0);
  ASSERT_EQ(x, y);
}

TEST(AlignedVector64, move_assignment) {
  AlignedVector64<uint64_t> x{1, 2, 3, 4};
  AlignedVector64<uint64_t> y = std::move(x);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(x.data()) % 64, 0);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(y.data()) % 64, 0);
  ASSERT_EQ(y, (AlignedVector64<uint64_t>{1, 2, 3, 4}));
}

TEST(AlignedVector64, copy_constructor) {
  AlignedVector64<uint64_t> x{1, 2, 3, 4};
  AlignedVector64<uint64_t> y{x};
  ASSERT_EQ(reinterpret_cast<uintptr_t>(x.data()) % 64, 0);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(y.data()) % 64, 0);
  ASSERT_EQ(y, (AlignedVector64<uint64_t>{1, 2, 3, 4}));
}

TEST(AlignedVector64, move_constructor) {
  AlignedVector64<uint64_t> x{1, 2, 3, 4};
  AlignedVector64<uint64_t> y{std::move(x)};
  ASSERT_EQ(reinterpret_cast<uintptr_t>(x.data()) % 64, 0);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(y.data()) % 64, 0);
  ASSERT_EQ(y, (AlignedVector64<uint64_t>{1, 2, 3, 4}));
}

}  // namespace hexl
}  // namespace intel
