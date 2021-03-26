// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

int main() {
  __m512i zero = _mm512_set1_epi64(0);
  __m512i one = _mm512_set1_epi64(1);
  __m512i two = _mm512_set1_epi64(2);
  __m512i out = _mm512_madd52lo_epu64(zero, one, two);
  __m256i out0 = _mm512_extracti64x4_epi64(out, 0);
  int result = _mm256_extract_epi64(out0, 0);
  int expected = 2;
  return (result == expected) ? 0 : 1;
}
