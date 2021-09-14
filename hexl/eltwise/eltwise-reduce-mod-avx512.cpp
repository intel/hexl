// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-reduce-mod-avx512.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

template void EltwiseReduceModAVX512<64>(uint64_t* result,
                                         const uint64_t* operand, uint64_t n,
                                         uint64_t modulus,
                                         uint64_t input_mod_factor,
                                         uint64_t output_mod_factor);
#endif

#ifdef HEXL_HAS_AVX512IFMA
template void EltwiseReduceModAVX512<52>(uint64_t* result,
                                         const uint64_t* operand, uint64_t n,
                                         uint64_t modulus,
                                         uint64_t input_mod_factor,
                                         uint64_t output_mod_factor);
#endif

}  // namespace hexl
}  // namespace intel
