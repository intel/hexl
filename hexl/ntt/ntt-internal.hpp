// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include <utility>

#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/util.hpp"
#include "util/util-internal.hpp"

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
void ForwardTransformToBitReverseRadix2(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor = 1,
    uint64_t output_mod_factor = 1);

/// @brief Radix-4 native C++ NTT implementation of the forward NTT
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
void ForwardTransformToBitReverseRadix4(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor = 1,
    uint64_t output_mod_factor = 1);

/// @brief Reference forward NTT which is written for clarity rather than
/// performance
/// @param[in, out] operand Input data. Overwritten with NTT output
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] modulus Prime modulus. Must satisfy q == 1 mod 2n
/// @param[in] root_of_unity_powers Powers of 2n'th root of unity in F_q. In
/// bit-reversed order
void ReferenceForwardTransformToBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers);

/// @brief Reference inverse NTT which is written for clarity rather than
/// performance
/// @param[in, out] operand Input data. Overwritten with NTT output
/// @param[in] n Size of the transform, i.e. the polynomial degree. Must be a
/// power of two.
/// @param[in] modulus Prime modulus. Must satisfy q == 1 mod 2n
/// @param[in] inv_root_of_unity_powers Powers of inverse 2n'th root of unity in
/// F_q. In bit-reversed order.
void ReferenceInverseTransformFromBitReverse(
    uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers);

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
void InverseTransformFromBitReverseRadix2(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers,
    uint64_t input_mod_factor = 1, uint64_t output_mod_factor = 1);

/// @brief Radix-4 native C++ NTT implementation of the inverse NTT
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
void InverseTransformFromBitReverseRadix4(
    uint64_t* result, const uint64_t* operand, uint64_t n, uint64_t modulus,
    const uint64_t* root_of_unity_powers,
    const uint64_t* precon_root_of_unity_powers, uint64_t input_mod_factor = 1,
    uint64_t output_mod_factor = 1);

}  // namespace hexl
}  // namespace intel
