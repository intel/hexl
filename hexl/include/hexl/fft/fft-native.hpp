// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <complex>

namespace intel {
namespace hexl {

/// @brief Radix-2 native C++ NTT implementation of the forward NTT
/// @param[out] result Output data. Overwritten with NTT output
/// @param[in] operand Input data.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] modulus Prime modulus q. Must satisfy q == 1 mod 2n
/// @param[in] root_of_unity_powers Powers of 2n'th root of unity in F_q. In
/// bit-reversed order
/// @param[in] precon_root_of_unity_powers Pre-conditioned Powers of 2n'th root
/// of unity in F_q. In bit-reversed order.
/// @param[in] input_mod_factor Upper bound for inputs; inputs must be in [0,
/// input_mod_factor * q)
/// @param[in] output_mod_factor Upper bound for result; result must be in [0,
/// output_mod_factor * q)
void Forward_FFT_ToBitReverseRadix2(
    std::complex<double_t>* result, const std::complex<double_t>* operand,
    const std::complex<double_t>* root_of_unity_powers, const uint64_t n,
    const double_t* scalar = nullptr);

/// @brief Radix-2 native C++ NTT implementation of the inverse NTT
/// @param[out] result Output data. Overwritten with NTT output
/// @param[in] operand Input data.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] modulus Prime modulus q. Must satisfy q == 1 mod 2n
/// @param[in] inv_root_of_unity_powers Powers of inverse 2n'th root of unity in
/// F_q. In bit-reversed order.
/// @param[in] precon_root_of_unity_powers Pre-conditioned powers of inverse
/// 2n'th root of unity in F_q. In bit-reversed order.
/// @param[in] input_mod_factor Upper bound for inputs; inputs must be in [0,
/// input_mod_factor * q)
/// @param[in] output_mod_factor Upper bound for result; result must be in [0,
/// output_mod_factor * q)
void Inverse_FFT_FromBitReverseRadix2(
    std::complex<double_t>* result, const std::complex<double_t>* operand,
    const std::complex<double_t>* inv_root_of_unity_powers, const uint64_t n,
    const double_t* scalar);

}  // namespace hexl
}  // namespace intel
