// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>

namespace intel {
namespace hexl {

/// @brief Radix-2 native C++ FFT implementation of the forward FFT
/// @param[out] result Output data. Overwritten with FFT output
/// @param[in] operand Input data.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] root_of_unity_powers Powers of 2n'th root of unity. In
/// bit-reversed order
/// @param[in] scale Scale applied to output data
void Forward_FFT_ToBitReverseRadix2(
    std::complex<double_t>* result, const std::complex<double_t>* operand,
    const std::complex<double_t>* root_of_unity_powers, const uint64_t n,
    const double_t* scale = nullptr);

/// @brief Radix-2 native C++ FFT implementation of the inverse FFT
/// @param[out] result Output data. Overwritten with FFT output
/// @param[in] operand Input data.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] inv_root_of_unity_powers Powers of inverse 2n'th root of unity.
/// In bit-reversed order.
/// @param[in] scale Scale applied to output data
void Inverse_FFT_FromBitReverseRadix2(
    std::complex<double_t>* result, const std::complex<double_t>* operand,
    const std::complex<double_t>* inv_root_of_unity_powers, const uint64_t n,
    const double_t* scale = nullptr);

}  // namespace hexl
}  // namespace intel
