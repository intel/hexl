// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace intel {
namespace hexl {

/// @brief Computes transposed linear regression
/// @param[in,out] result Ciphertext data. Will be over-written with result. Has
/// (3 * n * num_moduli) elements
/// @param[in] operand1 Vector of ciphertext representing a matrix that encodes
/// a transposed logistic regression model. Has (num_weights * 2 * n *
/// num_moduli) elements.
/// @param[in] operand2 Vector of ciphertext representing a matrix that encodes
/// at most n/2 input samples with feature size num_weights. Has (num_weights *
/// 2 * n * num_moduli) elements.
/// @param[in] n Number of coefficients in each polynomial
/// @param[in] moduli Pointer to contiguous array of num_moduli word-sized
/// coefficient moduli
/// @param[in] num_moduli Number of word-sized coefficient moduli
/// @param[in] num_weights Feature size of the linear/logistic regression model
void LinRegMatrixVectorMultiply(uint64_t* result, const uint64_t* operand1,
                                const uint64_t* operand2, uint64_t n,
                                const uint64_t* moduli, uint64_t num_moduli,
                                uint64_t num_weights);

}  // namespace hexl
}  // namespace intel
