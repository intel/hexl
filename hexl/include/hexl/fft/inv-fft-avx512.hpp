// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "hexl/fft/fft.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief AVX512 implementation of the forward NTT
/// @param[in, out] operand Input data. Overwritten with NTT output
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] modulus Prime modulus q. Must satisfy q == 1 mod 2n
/// @param[in] root_of_unity_powers Powers of 2n'th root of unity in F_q. In
/// bit-reversed order.
/// @param[in] precon_root_of_unity_powers Pre-conditioned Powers of 2n'th root
/// of unity in F_q. In bit-reversed order.
/// @param[in] input_mod_factor Upper bound for inputs; inputs must be in [0,
/// input_mod_factor * q)
/// @param[in] output_mod_factor Upper bound for result; result must be in [0,
/// output_mod_factor * q)
/// @param[in] recursion_depth Depth of recursive call
/// @param[in] recursion_half Helper for indexing roots of unity
/// @details The implementation is recursive. The base case is a breadth-first
/// NTT, where all the butterflies in a given stage are processed before any
/// butterflies in the next stage. The base case is small enough to fit in the
/// smallest cache. Larger NTTs are processed recursively in a depth-first
/// manner, such that an entire subtransform is completed before moving to the
/// next subtransform. This reduces the number of cache misses, improving
/// performance on larger transform sizes.
void InvFFTFromBitReverseAVX512(double_t* result_8C_intrlvd,
                                const double_t* operand_8C_intrlvd,
                                const double_t* root_of_unity_powers_1C_intrlvd,
                                const uint64_t n,
                                const double_t* scalar = nullptr);

void BuildFloatingPointsAVX512(double_t* res, const uint64_t* plain,
                               const uint64_t* threshold,
                               const uint64_t* decryption_modulus,
                               const double_t inv_scale, const size_t mod_size,
                               const size_t coeff_count);

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
