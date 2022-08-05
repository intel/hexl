// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef HEXL_FPGA_COMPATIBLE_DYADIC_MULTIPLY

#include "hexl/experimental/seal/dyadic-multiply.hpp"

#include "hexl/experimental/seal/dyadic-multiply-internal.hpp"

namespace intel {
namespace hexl {

void DyadicMultiply(uint64_t* result, const uint64_t* operand1,
                    const uint64_t* operand2, uint64_t n,
                    const uint64_t* moduli, uint64_t num_moduli) {
  intel::hexl::internal::DyadicMultiply(result, operand1, operand2, n, moduli,
                                        num_moduli);
}

}  // namespace hexl
}  // namespace intel
#endif
