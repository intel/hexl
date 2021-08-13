// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test-util-avx512.hpp"
#include "util/avx512ifma-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512IFMA
TEST(AVX512, _mm512_hexl_mulhi_epi52) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }
  __m512i x = _mm512_set_epi64(90774764920991, 90774764920991, 90774764920991,
                               90774764920991, 90774764920991, 90774764920991,
                               90774764920991, 90774764920991);
  __m512i y = _mm512_set_epi64(424, 635, 757, 457, 280, 624, 353, 496);

  __m512i expected = _mm512_set_epi64(8, 12, 15, 9, 5, 12, 7, 9);

  __m512i z = _mm512_hexl_mulhi_epi<52>(x, y);

  CheckEqual(z, expected);
}
#endif

}  // namespace hexl
}  // namespace intel
