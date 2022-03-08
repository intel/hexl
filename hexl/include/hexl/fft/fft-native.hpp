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
void Forward_FFT_Radix2(std::complex<double>* result,
                        const std::complex<double>* operand,
                        const std::complex<double>* root_of_unity_powers,
                        const uint64_t n);

/// @brief Radix-2 native C++ FFT implementation of the inverse FFT
/// @param[out] result Output data. Overwritten with FFT output
/// @param[in] operand Input data.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] inv_root_of_unity_powers Powers of inverse 2n'th root of unity.
/// In bit-reversed order.
void Inverse_FFT_Radix2(std::complex<double>* result,
                        const std::complex<double>* operand,
                        const std::complex<double>* inv_root_of_unity_powers,
                        const uint64_t n);

}  // namespace hexl
}  // namespace intel
