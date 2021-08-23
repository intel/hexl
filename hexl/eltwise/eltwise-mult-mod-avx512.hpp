// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "eltwise/eltwise-mult-mod-internal.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/number-theory/number-theory.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ

/// @brief Multiplies two vectors elementwise with modular reduction
/// @param[in] result Result of element-wise multiplication
/// @param[in] operand1 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] operand2 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * p) Must be 1, 2 or 4.
/// @details Computes \p result[i] = (\p operand1[i] * \p operand2[i]) mod \p
/// modulus for i=0, ..., \p n - 1
/// @details Barrett's algorithm for vector-vector modular multiplication
/// (Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document)
/// using AVX512IFMA
template <int InputModFactor>
void EltwiseMultModAVX512IFMAInt(uint64_t* result, const uint64_t* operand1,
                                 const uint64_t* operand2, uint64_t n,
                                 uint64_t modulus);

/// @brief Multiplies two vectors elementwise with modular reduction
/// @param[in] result Result of element-wise multiplication
/// @param[in] operand1 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] operand2 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * p) Must be 1, 2 or 4.
/// @details Computes \p result[i] = (\p operand1[i] * \p operand2[i]) mod \p
/// modulus for i=0, ..., \p n - 1
/// @details Barrett's algorithm for vector-vector modular multiplication
/// (Algorithm 1 from https://hal.archives-ouvertes.fr/hal-01215845/document)
/// using AVX512DQ
template <int InputModFactor>
void EltwiseMultModAVX512DQInt(uint64_t* result, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               uint64_t modulus);

/// @brief Multiplies two vectors elementwise with modular reduction
/// @param[in] result Result of element-wise multiplication
/// @param[in] operand1 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] operand2 Vector of elements to multiply. Each element must be
/// less than the modulus.
/// @param[in] n Number of elements in each vector
/// @param[in] modulus Modulus with which to perform modular reduction
/// @param[in] input_mod_factor Assumes input elements are in [0,
/// input_mod_factor * p) Must be 1, 2 or 4.
/// @details Computes \p result[i] = (\p operand1[i] * \p operand2[i]) mod \p
/// modulus for i=0, ..., \p n - 1
/// @details Function 18 on page 19 of https://arxiv.org/pdf/1407.3383.pdf
/// See also Algorithm 2/3 of
/// https://hal.archives-ouvertes.fr/hal-02552673/document
/// Uses floating-point arithmetic
template <int InputModFactor>
void EltwiseMultModAVX512Float(uint64_t* result, const uint64_t* operand1,
                               const uint64_t* operand2, uint64_t n,
                               uint64_t modulus);

#endif  // HEXL_HAS_AVX512DQ

}  // namespace hexl
}  // namespace intel
