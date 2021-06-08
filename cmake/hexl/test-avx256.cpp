// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

int main() {
  __m256i one = _mm256_set1_epi64x(1);
  __m256i two = _mm256_set1_epi64x(2);
  __m256i sum = _mm256_add_epi64(one, two);
  int result = _mm256_extract_epi64(sum, 0);
  int expected = 3;
  return (result == expected) ? 0 : 1;
}
