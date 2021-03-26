// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "eltwise/eltwise-mult-mod-avx512.hpp"

#include <immintrin.h>
#include <stdint.h>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "intel-hexl/eltwise/eltwise-mult-mod.hpp"
#include "number-theory/number-theory.hpp"
#include "util/avx512-util.hpp"
#include "util/check.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

template void EltwiseMultModAVX512Float<1>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);
template void EltwiseMultModAVX512Float<2>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);
template void EltwiseMultModAVX512Float<4>(uint64_t* result,
                                           const uint64_t* operand1,
                                           const uint64_t* operand2, uint64_t n,
                                           uint64_t modulus);

template void EltwiseMultModAVX512Int<1>(uint64_t* result,
                                         const uint64_t* operand1,
                                         const uint64_t* operand2, uint64_t n,
                                         uint64_t modulus);
template void EltwiseMultModAVX512Int<2>(uint64_t* result,
                                         const uint64_t* operand1,
                                         const uint64_t* operand2, uint64_t n,
                                         uint64_t modulus);
template void EltwiseMultModAVX512Int<4>(uint64_t* result,
                                         const uint64_t* operand1,
                                         const uint64_t* operand2, uint64_t n,
                                         uint64_t modulus);

#endif

}  // namespace hexl
}  // namespace intel
