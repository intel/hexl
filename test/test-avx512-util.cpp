// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include <vector>

#include "gtest/gtest.h"
#include "test-util-avx512.hpp"
#include "util/avx512-util.hpp"
#include "util/cpu-features.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

TEST(AVX512, ExtractValues) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  __m512i x = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);

  AssertEqual(ExtractValues(x), std::vector<uint64_t>{8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(AVX512, ExtractIntValues) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  __m512i x = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
  AssertEqual(ExtractIntValues(x),
              std::vector<int64_t>{8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(AVX512, ExtractDoubleValues) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  __m512d x = _mm512_set_pd(-4.4, -3.3, -2.2, -1.1, 0, 1.1, 2.2, 3.3);
  AssertEqual(ExtractValues(x),
              std::vector<double>{3.3, 2.2, 1.1, 0, -1.1, -2.2, -3.3, -4.4});
}
#endif

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

#ifdef HEXL_HAS_AVX512DQ
TEST(AVX512, _mm512_hexl_mulhi_epi64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  __m512i w = _mm512_set_epi64(90774764920991,    //
                               1ULL << 63,        //
                               1ULL << 63,        //
                               1ULL << 63,        //
                               1ULL << 63,        //
                               1ULL << 63,        //
                               (1ULL << 60) + 1,  //
                               (1ULL << 62) + 2);
  __m512i y = _mm512_set_epi64(1ULL << 63,        //
                               1ULL << 63,        //
                               (1ULL << 63) + 1,  //
                               (1ULL << 63) + 2,  //
                               (1ULL << 63) + 3,  //
                               (1ULL << 63) + 4,  //
                               (1ULL << 60) + 3,  //
                               (1ULL << 63) + 4);

  __m512i expected = _mm512_set_epi64(90774764920991 >> 1,  //
                                      1ULL << 62,           //
                                      1ULL << 62,           //
                                      (1ULL << 62) + 1,     //
                                      (1ULL << 62) + 1,     //
                                      (1ULL << 62) + 2,     //
                                      1ULL << 56,           //
                                      (1ULL << 61) + 2);

  {
    __m512i z = _mm512_hexl_mulhi_epi<64>(w, y);
    CheckEqual(z, expected);
  }

  {
    __m512i z = _mm512_hexl_mulhi_approx_epi<64>(w, y);
    CheckClose(z, expected, 1);
  }
}
#endif

#ifdef HEXL_HAS_AVX512DQ
TEST(AVX512, _mm512_hexl_cmplt_epu64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  // Small
  {
    uint64_t match_value = 10;
    __m512i a = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i b = _mm512_set_epi64(0, 1, 1, 0, 5, 6, 100, 100);
    __m512i expected_out = _mm512_set_epi64(
        0, 0, 0, 0, match_value, match_value, match_value, match_value);

    __m512i c = _mm512_hexl_cmplt_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }

  // Large
  {
    uint64_t match_value = 13;
    __m512i a = _mm512_set_epi64(1ULL << 32,         //
                                 1ULL << 63,         //
                                 (1ULL << 63) + 1,   //
                                 (1ULL << 63) + 10,  //
                                 0,                  //
                                 0,                  //
                                 0,                  //
                                 0);
    __m512i b = _mm512_set_epi64(1ULL << 32,         //
                                 1ULL << 63,         //
                                 1ULL << 63,         //
                                 (1ULL << 63) + 17,  //
                                 0,                  //
                                 0,                  //
                                 0,                  //
                                 0);
    __m512i expected_out = _mm512_set_epi64(0, 0, 0, match_value, 0, 0, 0, 0);

    __m512i c = _mm512_hexl_cmplt_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }
}

TEST(AVX512, _mm512_hexl_cmple_epu64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }
  // Small
  {
    uint64_t match_value = 10;
    __m512i a = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i b = _mm512_set_epi64(0, 1, 1, 0, 5, 6, 100, 100);
    __m512i expected_out =
        _mm512_set_epi64(match_value, match_value, 0, 0, match_value,
                         match_value, match_value, match_value);

    __m512i c = _mm512_hexl_cmple_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }

  // Large
  {
    uint64_t match_value = 13;
    __m512i a = _mm512_set_epi64(1ULL << 32,         //
                                 1ULL << 63,         //
                                 (1ULL << 63) + 1,   //
                                 (1ULL << 63) + 10,  //
                                 0,                  //
                                 0,                  //
                                 0,                  //
                                 0);
    __m512i b = _mm512_set_epi64(1ULL << 32,         //
                                 1ULL << 63,         //
                                 1ULL << 63,         //
                                 (1ULL << 63) + 17,  //
                                 0,                  //
                                 0,                  //
                                 0,                  //
                                 0);
    __m512i expected_out =
        _mm512_set_epi64(match_value, match_value, 0, match_value, match_value,
                         match_value, match_value, match_value);

    __m512i c = _mm512_hexl_cmple_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }
}

TEST(AVX512, _mm512_hexl_cmpge_epu64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  // Small
  {
    uint64_t match_value = 10;
    __m512i a = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    __m512i b = _mm512_set_epi64(0, 1, 1, 0, 5, 6, 100, 100);
    __m512i expected_out = _mm512_set_epi64(
        match_value, match_value, match_value, match_value, 0, 0, 0, 0);

    __m512i c = _mm512_hexl_cmpge_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }

  // Large
  {
    uint64_t match_value = 13;
    __m512i a = _mm512_set_epi64(1ULL << 32,         //
                                 1ULL << 63,         //
                                 (1ULL << 63) + 1,   //
                                 (1ULL << 63) + 10,  //
                                 0,                  //
                                 0,                  //
                                 0,                  //
                                 0);
    __m512i b = _mm512_set_epi64(1ULL << 32,         //
                                 1ULL << 63,         //
                                 1ULL << 63,         //
                                 (1ULL << 63) + 17,  //
                                 0,                  //
                                 0,                  //
                                 0,                  //
                                 0);
    __m512i expected_out =
        _mm512_set_epi64(match_value, match_value, match_value, 0, match_value,
                         match_value, match_value, match_value);

    __m512i c = _mm512_hexl_cmpge_epu64(a, b, match_value);

    CheckEqual(c, expected_out);
  }
}

TEST(AVX512, _mm512_hexl_small_mod_epu64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  // Small
  {
    __m512i a = _mm512_set_epi64(0, 2, 4, 6, 8, 10, 11, 12);
    __m512i moduli = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
    __m512i expected_out = _mm512_set_epi64(0, 0, 1, 2, 3, 4, 4, 4);

    __m512i c = _mm512_hexl_small_mod_epu64(a, moduli);

    CheckEqual(c, expected_out);
  }

  // Large
  {
    __m512i a = _mm512_set_epi64(1ULL << 32,         //
                                 1ULL << 63,         //
                                 (1ULL << 63) + 1,   //
                                 (1ULL << 63) + 10,  //
                                 0,                  //
                                 0,                  //
                                 0,                  //
                                 0);
    __m512i moduli = _mm512_set_epi64(1ULL << 32,         //
                                      1ULL << 63,         //
                                      1ULL << 63,         //
                                      (1ULL << 63) + 17,  //
                                      0,                  //
                                      0,                  //
                                      0,                  //
                                      0);
    __m512i expected_out =
        _mm512_set_epi64(0, 0, 1, (1ULL << 63) + 10, 0, 0, 0, 0);

    __m512i c = _mm512_hexl_small_mod_epu64(a, moduli);

    CheckEqual(c, expected_out);
  }
}

TEST(AVX512, _mm512_hexl_barrett_reduce64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  // Small
  {
    __m512i a = _mm512_set_epi64(12, 11, 10, 8, 6, 4, 2, 0);

    uint64_t modulus = 5;
    uint64_t barrett_factor = MultiplyFactor(1, 64, modulus).BarrettFactor();
    __m512i vmoduli = _mm512_set1_epi64(modulus);
    __m512i vbarrs = _mm512_set1_epi64(barrett_factor);

    // Multi-word Barrett reduction precomputation
    constexpr int64_t beta = -2;
    uint64_t ceil_log_mod = Log2(modulus) + 1;
    uint64_t prod_right_shift = ceil_log_mod + beta;
    __m512i v_neg_mod = _mm512_set1_epi64(-static_cast<int64_t>(modulus));

    __m512i expected_out = _mm512_set_epi64(2, 1, 0, 3, 1, 4, 2, 0);

    __m512i c = _mm512_hexl_barrett_reduce64(a, vmoduli, vbarrs, vbarrs,
                                             prod_right_shift, v_neg_mod);
    AssertEqual(c, expected_out);
  }

  // Random
  {
    uint64_t modulus = 75;
    __m512i vmodulus = _mm512_set1_epi64(modulus);
    __m512i vbarr =
        _mm512_set1_epi64(MultiplyFactor(1, 64, modulus).BarrettFactor());

    // Multi-word Barrett reduction precomputation
    constexpr int64_t beta = -2;
    const uint64_t ceil_log_mod = Log2(modulus) + 1;  // "n" from Algorithm 2
    uint64_t prod_right_shift = ceil_log_mod + beta;
    __m512i v_neg_mod = _mm512_set1_epi64(-static_cast<int64_t>(modulus));

    for (size_t trial = 0; trial < 200; ++trial) {
      auto arg1 = GenerateInsecureUniformRandomValues(8, 0, modulus * modulus);
      auto exp = arg1;
      for (auto& elem : exp) {
        elem %= modulus;
      }

      __m512i varg1 = _mm512_set_epi64(arg1[7], arg1[6], arg1[5], arg1[4],
                                       arg1[3], arg1[2], arg1[1], arg1[0]);

      __m512i c = _mm512_hexl_barrett_reduce64(varg1, vmodulus, vbarr, vbarr,
                                               prod_right_shift, v_neg_mod);
      std::vector<uint64_t> result = ExtractValues(c);

      AssertEqual(result, exp);
    }
  }
}
#endif

}  // namespace hexl
}  // namespace intel
