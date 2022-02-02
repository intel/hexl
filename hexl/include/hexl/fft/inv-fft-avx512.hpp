// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/fft/fft.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief AVX512 implementation of the inverse FFT
/// @param[out] result_cmplx_intrlvd Output data. Overwritten with FFT output.
/// Result is a vector of double with interleaved real and imaginary numbers.
/// @param[in] operand_cmplx_intrlvd Input data. A vector of double with
/// interleaved real and imaginary numbers.
/// @param[in] inv_roots_of_unity_cmplx_intrlvd Powers of 2n'th root of unity.
/// In bit-reversed order.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] scale Scale applied to output values
void Inverse_FFT_FromBitReverseAVX512(
    double_t* result_cmplx_intrlvd, const double_t* operand_cmplx_intrlvd,
    const double_t* inv_root_of_unity_cmplxintrlvd, const uint64_t n,
    const double_t* scale = nullptr, uint64_t recursion_depth = 0,
    uint64_t recursion_half = 0);

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
