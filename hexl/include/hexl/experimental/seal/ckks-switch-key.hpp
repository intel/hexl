// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

namespace intel {
namespace hexl {

/// @brief Computes CKKS key switching in-place
/// @param[in,out] result Ciphertext data. Will be over-written with result. Has
/// (n * decomp_modulus_size * key_component_count) elements
/// @param[in] t_target_iter_ptr TODO(fboemer)
/// @param[in] n Number of coefficients in each polynomial
/// @param[in] decomp_modulus_size  Number of moduli in the ciphertext at its
/// current level, excluding one auxiliary prime.
/// @param[in] key_modulus_size Number of moduli in the ciphertext at its top
/// level, including one auxiliary prime.
/// @param[in] rns_modulus_size Number of moduli in the ciphertext at its
/// current level, including one auxiliary prime. rns_modulus_size ==
/// decomp_modulus_size + 1
/// @param[in] key_component_count TODO(fboemer)
/// @param[in] moduli Array of word-sized coefficient moduli. There must be
/// key_modulus_size moduli in the array
/// @param[in] kswitch_keys Array of evaluation key data. Has
/// decomp_modulus_size entries, each with
/// coeff_count * ((key_modulus_size - 1)+ (key_component_count - 1) *
/// (key_modulus_size) + 1) entries
/// @param[in] modswitch_factors Array of modulus switch factors
void CkksSwitchKey(uint64_t* result, const uint64_t* t_target_iter_ptr,
                   uint64_t n, uint64_t decomp_modulus_size,
                   uint64_t key_modulus_size, uint64_t rns_modulus_size,
                   uint64_t key_component_count, uint64_t* moduli,
                   const uint64_t** kswitch_keys, uint64_t* modswitch_factors);

}  // namespace hexl
}  // namespace intel
