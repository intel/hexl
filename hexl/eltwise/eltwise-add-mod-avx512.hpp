// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

void EltwiseAddModAVX512(uint64_t* result, const uint64_t* operand1,
                         const uint64_t* operand2, uint64_t n,
                         uint64_t modulus);

void EltwiseAddModAVX512(uint64_t* result, const uint64_t* operand1,
                         const uint64_t operand2, uint64_t n, uint64_t modulus);

}  // namespace hexl
}  // namespace intel
