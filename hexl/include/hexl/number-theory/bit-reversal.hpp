// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

enum class BitOrdering { Normal, BitReversed };

void BitReversal(uint64_t* input, uint64_t size);

}  // namespace hexl
}  // namespace intel
