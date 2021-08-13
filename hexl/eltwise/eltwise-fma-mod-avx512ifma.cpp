// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include "eltwise/eltwise-fma-mod-avx512.tpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512IFMA
template void EltwiseFMAModAVX512<52, 1>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<52, 2>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<52, 4>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
template void EltwiseFMAModAVX512<52, 8>(uint64_t* result, const uint64_t* arg1,
                                         uint64_t arg2, const uint64_t* arg3,
                                         uint64_t n, uint64_t modulus);
#endif

}  // namespace hexl
}  // namespace intel
