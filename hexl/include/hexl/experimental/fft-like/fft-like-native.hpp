// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>

namespace intel {
namespace hexl {

/// @brief Radix-2 native C++ FFT like implementation of the forward FFT like
/// @param[out] result Output data. Overwritten with FFT like output
/// @param[in] operand Input data.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] root_of_unity_powers Powers of 2n'th root of unity. In
/// bit-reversed order
/// @param[in] scale Scale applied to output data
void Forward_FFTLike_ToBitReverseRadix2(
    std::complex<double>* result, const std::complex<double>* operand,
    const std::complex<double>* root_of_unity_powers, const uint64_t n,
    const double* scale = nullptr);

/// @brief Radix-2 native C++ FFT like implementation of the inverse FFT like
/// @param[out] result Output data. Overwritten with FFT like output
/// @param[in] operand Input data.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] inv_root_of_unity_powers Powers of inverse 2n'th root of unity.
/// In bit-reversed order.
/// @param[in] scale Scale applied to output data
void Inverse_FFTLike_FromBitReverseRadix2(
    std::complex<double>* result, const std::complex<double>* operand,
    const std::complex<double>* inv_root_of_unity_powers, const uint64_t n,
    const double* scale = nullptr);

}  // namespace hexl
}  // namespace intel
