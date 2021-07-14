// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <immintrin.h>
#include <stdint.h>

#include <limits>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"
#include "util/avx512-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

// Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document
template <int InputModFactor>
void EltwiseMultModAVX512Int(uint64_t* result, const uint64_t* operand1,
                             const uint64_t* operand2, uint64_t n,
                             uint64_t modulus);

template <int InputModFactor>
void EltwiseMultModAVX512Float(uint64_t* result, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               uint64_t modulus);

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
