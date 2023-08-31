// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#ifndef HEXL_FPGA_COMPATIBLE_KEYSWITCH

#include "hexl/experimental/seal/key-switch.hpp"

#include "hexl/experimental/seal/key-switch-internal.hpp"

namespace intel {
namespace hexl {

void KeySwitch(uint64_t* result, const uint64_t* t_target_iter_ptr, uint64_t n,
               uint64_t decomp_modulus_size, uint64_t key_modulus_size,
               uint64_t rns_modulus_size, uint64_t key_component_count,
               const uint64_t* moduli, const uint64_t** k_switch_keys,
               const uint64_t* modswitch_factors,
               const uint64_t* root_of_unity_powers_ptr) {
  intel::hexl::internal::KeySwitch(
      result, t_target_iter_ptr, n, decomp_modulus_size, key_modulus_size,
      rns_modulus_size, key_component_count, moduli, k_switch_keys,
      modswitch_factors, root_of_unity_powers_ptr);
}

}  // namespace hexl
}  // namespace intel
#endif
