// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "hexl/experimental/fft-like/fft-like.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief AVX512 implementation of the forward FFT like
/// @param[out] result_cmplx_intrlvd Output data. Overwritten with FFT like
/// output. Result is a vector of double with interleaved real and imaginary
/// numbers.
/// @param[in] operand_cmplx_intrlvd Input data. A vector of double with
/// interleaved real and imaginary numbers.
/// @param[in] roots_of_unity_cmplx_intrlvd Powers of 2n'th root of unity. In
/// bit-reversed order.
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] scale Scale applied to output values
/// @param[in] recursion_depth Depth of recursive call
/// @param[in] recursion_half Helper for indexing roots of unity
/// @details The implementation is recursive. The base case is a breadth-first
/// FFT like, where all the butterflies in a given stage are processed before
/// any butterflies in the next stage. The base case is small enough to fit in
/// the smallest cache. Larger FFTs are processed recursively in a depth-first
/// manner, such that an entire subtransform is completed before moving to the
/// next subtransform. This reduces the number of cache misses, improving
/// performance on larger transform sizes.
void Forward_FFTLike_ToBitReverseAVX512(
    double* result_cmplx_intrlvd, const double* operand_cmplx_intrlvd,
    const double* roots_of_unity_cmplx_intrlvd, const uint64_t n,
    const double* scale = nullptr, uint64_t recursion_depth = 0,
    uint64_t recursion_half = 0);

/// @brief Construct floating-point values from CRT-composed polynomial with
/// integer coefficients in AVX512.
/// @param[out] res_cmplx_intrlvd Stores the result
/// @param[in] plain Plaintext
/// @param[in] threshold Upper half threshold with respect to the total
/// coefficient modulus
/// @param[in] decryption_modulus Product of all primes in the coefficient
/// modulus
/// @param[in] inv_scale Scale applied to output values
/// @param[in] mod_size Size of coefficient modulus parameter
/// @param[in] coeff_count Degree of the polynomial modulus parameter
void BuildFloatingPointsAVX512(double* res_cmplx_intrlvd, const uint64_t* plain,
                               const uint64_t* threshold,
                               const uint64_t* decryption_modulus,
                               const double inv_scale, const size_t mod_size,
                               const size_t coeff_count);

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
