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
void Inverse_FFT_AVX512(double* result_cmplx_intrlvd,
                        const double* operand_cmplx_intrlvd,
                        const double* inv_root_of_unity_cmplxintrlvd,
                        const size_t* rev_idx, const size_t* idx_rev,
                        const uint64_t n);

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
