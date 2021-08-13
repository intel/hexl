// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>
#include <stdint.h>

#include <limits>

#include "eltwise/eltwise-mult-mod-avx512.hpp"
#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512VBMI2

template void EltwiseMultModAVX512VBMI2Int<1>(uint64_t* result,
                                              const uint64_t* operand1,
                                              const uint64_t* operand2,
                                              uint64_t n, uint64_t modulus);
template void EltwiseMultModAVX512VBMI2Int<2>(uint64_t* result,
                                              const uint64_t* operand1,
                                              const uint64_t* operand2,
                                              uint64_t n, uint64_t modulus);
template void EltwiseMultModAVX512VBMI2Int<4>(uint64_t* result,
                                              const uint64_t* operand1,
                                              const uint64_t* operand2,
                                              uint64_t n, uint64_t modulus);

template <int InputModFactor>
void EltwiseMultModAVX512VBMI2Int(uint64_t* result, const uint64_t* operand1,
                                  const uint64_t* operand2, uint64_t n,
                                  uint64_t modulus) {
  EltwiseMultModAVX512Int<InputModFactor>(result, operand1, operand2, n,
                                          modulus);
}

}  // namespace hexl
}  // namespace intel
