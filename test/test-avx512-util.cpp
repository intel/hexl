// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include <memory>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test-util.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

TEST(AVX512, ExtractValues) {
  __m512i x = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);

  AssertEqual(ExtractValues(x), std::vector<uint64_t>{8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(AVX512, ExtractIntValues) {
  __m512i x = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
  AssertEqual(ExtractIntValues(x),
              std::vector<int64_t>{8, 7, 6, 5, 4, 3, 2, 1});
}

TEST(AVX512, ExtractDoubleValues) {
  __m512d x = _mm512_set_pd(-4.4, -3.3, -2.2, -1.1, 0, 1.1, 2.2, 3.3);
  AssertEqual(ExtractDoubleValues(x),
              std::vector<double>{3.3, 2.2, 1.1, 0, -1.1, -2.2, -3.3, -4.4});
}
#endif

#ifdef HEXL_HAS_AVX512IFMA
TEST(AVX512, _mm512_hexl_mulhi_epi52) {
  __m512i w = _mm512_set_epi64(90774764920991, 90774764920991, 90774764920991,
                               90774764920991, 90774764920991, 90774764920991,
                               90774764920991, 90774764920991);
  __m512i y = _mm512_set_epi64(424, 635, 757, 457, 280, 624, 353, 496);

  __m512i expected = _mm512_set_epi64(8, 12, 15, 9, 5, 12, 7, 9);

  __m512i z = _mm512_hexl_mulhi_epi<52>(w, y);

  CheckEqual(z, expected);
}
#endif

#ifdef HEXL_HAS_AVX512DQ
TEST(AVX512, _mm512_hexl_cmplt_epu64) {
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
  // Small
  {
    __m512i a = _mm512_set_epi64(12, 11, 10, 8, 6, 4, 2, 0);

    std::vector<uint64_t> moduli{2, 2, 3, 4, 5, 6, 7, 8};
    std::vector<uint64_t> barrs(moduli.size());
    for (size_t i = 0; i < barrs.size(); ++i) {
      barrs[i] = MultiplyFactor(1, 64, moduli[i]).BarrettFactor();
    }

    __m512i vmoduli =
        _mm512_set_epi64(moduli[7], moduli[6], moduli[5], moduli[4], moduli[3],
                         moduli[2], moduli[1], moduli[0]);
    __m512i vbarrs = _mm512_set_epi64(barrs[7], barrs[6], barrs[5], barrs[4],
                                      barrs[3], barrs[2], barrs[1], barrs[0]);

    __m512i expected_out = _mm512_set_epi64(4, 4, 4, 3, 2, 1, 0, 0);

    __m512i c = _mm512_hexl_barrett_reduce64(a, vmoduli, vbarrs);
    AssertEqual(c, expected_out);
  }

  // Random
  {
    std::random_device rd;
    std::mt19937 gen(rd());

    uint64_t modulus = 75;
    std::uniform_int_distribution<uint64_t> distrib(50, modulus * modulus - 1);
    __m512i vmodulus = _mm512_set1_epi64(modulus);
    __m512i vbarr =
        _mm512_set1_epi64(MultiplyFactor(1, 64, modulus).BarrettFactor());

    for (size_t trial = 0; trial < 200; ++trial) {
      std::vector<uint64_t> arg1(8, 0);
      std::vector<uint64_t> exp(8, 0);
      for (size_t i = 0; i < 8; ++i) {
        arg1[i] = distrib(gen);
        exp[i] = arg1[i] % modulus;
      }
      __m512i varg1 = _mm512_set_epi64(arg1[7], arg1[6], arg1[5], arg1[4],
                                       arg1[3], arg1[2], arg1[1], arg1[0]);

      __m512i c = _mm512_hexl_barrett_reduce64(varg1, vmodulus, vbarr);
      std::vector<uint64_t> result = ExtractValues(c);

      ASSERT_EQ(result, exp);
    }
  }
}

TEST(AVX512, _mm512_hexl_add_epi128) {
  // Small
  {
    __m512i a = _mm512_set_epi64(0, 1, 0, 2, 0, 3, 0, 4);
    __m512i b = _mm512_set_epi64(0, 5, 0, 6, 0, 7, 0, 8);

    __m512i expected_out = _mm512_set_epi64(0, 6, 0, 8, 0, 10, 0, 12);

    __m512i out = _mm512_hexl_add_epi128(a, b);

    AssertEqual(out, expected_out);
  }

  // Big
  {
    __m512i a = _mm512_set_epi64(1, 1, 2, 2, 3, 3, 4, 4);
    __m512i b = _mm512_set_epi64(5, 5, 6, 6, 7, 7, 8, 8);

    __m512i expected_out = _mm512_set_epi64(6, 6, 8, 8, 10, 10, 12, 12);

    __m512i out = _mm512_hexl_add_epi128(a, b);

    AssertEqual(out, expected_out);
  }

  // Overflow
  {
    __m512i a = _mm512_set_epi64(1, (1ULL << 63),      //
                                 1, (1ULL << 63) + 1,  //
                                 1, (1ULL << 63) + 1,  //
                                 0, 0);
    __m512i b = _mm512_set_epi64(5, (1ULL << 63),      //
                                 5, (1ULL << 63) + 6,  //
                                 0, 7,                 //
                                 0, 0);

    __m512i expected_out = _mm512_set_epi64(7, 0,                 //
                                            7, 7,                 //
                                            1, (1ULL << 63) + 8,  //
                                            0, 0);

    __m512i out = _mm512_hexl_add_epi128(a, b);

    AssertEqual(out, expected_out);
  }
}

#endif

}  // namespace hexl
}  // namespace intel
