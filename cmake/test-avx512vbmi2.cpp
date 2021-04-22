// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

int main() {
  __m512i high_bits = _mm512_set1_epi64(1);
  __m512i low_bits = _mm512_set1_epi64(0);

  __m512i shift = _mm512_shrdi_epi64(low_bits, high_bits, 60);
  __m256i shift0 = _mm512_extracti64x4_epi64(shift, 0);
  int result = _mm256_extract_epi64(shift0, 0);
  int expected = 16;  // 2**64 / 2**60

  return (result == expected) ? 0 : 1;
}
