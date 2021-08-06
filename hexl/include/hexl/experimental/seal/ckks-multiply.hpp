// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace intel {
namespace hexl {

void CKKSMultiply(uint64_t* result, const uint64_t* operand1,
                  const uint64_t* operand2, uint64_t n, const uint64_t* moduli,
                  uint64_t num_moduli);

}  // namespace hexl
}  // namespace intel
