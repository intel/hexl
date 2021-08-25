// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace intel {
namespace hexl {

/// @brief Computes CKKS multiplication
/// @param[in,out] result Ciphertext data. Will be over-written with result. Has
/// (2 * n * num_moduli) elements
/// @param[in] operand1 First ciphertext argument. Has (2 * n * num_moduli)
/// elements.
/// @param[in] operand2 Second ciphertext argument. Has (2 * n * num_moduli)
/// elements.
/// @param[in] n Number of coefficients in each polynomial
/// @param[in] moduli Pointer to contiguous array of num_moduli word-sized
/// coefficient moduli
/// @param[in] num_moduli Number of word-sized coefficient moduli
void CkksMultiply(uint64_t* result, const uint64_t* operand1,
                  const uint64_t* operand2, uint64_t n, const uint64_t* moduli,
                  uint64_t num_moduli);

}  // namespace hexl
}  // namespace intel
