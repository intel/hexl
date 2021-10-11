// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#include "hexl/eltwise/eltwise-add-mod.hpp"
#include "hexl/eltwise/eltwise-fma-mod.hpp"
#include "hexl/eltwise/eltwise-mult-mod.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void CkksSwitchKey(uint64_t* result, const uint64_t* t_target_iter_ptr,
                   uint64_t n, uint64_t decomp_modulus_size,
                   uint64_t key_modulus_size, uint64_t rns_modulus_size,
                   uint64_t key_component_count, uint64_t* moduli,
                   const uint64_t** k_switch_keys,
                   uint64_t* modswitch_factors) {
  uint64_t coeff_count = n;

  // Create a copy of target_iter
  std::vector<uint64_t> t_target(coeff_count * decomp_modulus_size, 0);
  for (size_t i = 0; i < coeff_count * decomp_modulus_size; ++i) {
    t_target[i] = t_target_iter_ptr[i];
  }

  uint64_t* t_target_ptr = &t_target[0];

  // Simplified implementation, where we assume no modular reduction is required
  // for intermediate additions
  std::vector<uint64_t> t_ntt(coeff_count, 0);
  uint64_t* t_ntt_ptr = &t_ntt[0];

  // In CKKS t_target is in NTT form; switch
  // back to normal form
  for (size_t j = 0; j < decomp_modulus_size; ++j) {
    NTT(n, moduli[j])
        .ComputeInverse(&t_target_ptr[j * coeff_count],
                        &t_target_ptr[j * coeff_count], 2, 1);
  }

  std::vector<uint64_t> t_poly_prod(
      key_component_count * coeff_count * rns_modulus_size, 0);

  for (size_t i = 0; i < rns_modulus_size; ++i) {
    size_t key_index = (i == decomp_modulus_size ? key_modulus_size - 1 : i);

    // Allocate memory for a lazy accumulator (128-bit coefficients)
    std::vector<uint64_t> t_poly_lazy(key_component_count * coeff_count * 2, 0);
    uint64_t* t_poly_lazy_ptr = &t_poly_lazy[0];
    uint64_t* accumulator_ptr = &t_poly_lazy[0];

    for (size_t j = 0; j < decomp_modulus_size; ++j) {
      const uint64_t* t_operand;
      // assume scheme == scheme_type::ckks
      if (i == j) {
        t_operand = &t_target_iter_ptr[j * coeff_count];
      } else {
        // Perform RNS-NTT conversion
        // No need to perform RNS conversion (modular reduction)
        if (moduli[j] <= moduli[key_index]) {
          for (size_t l = 0; l < coeff_count; ++l) {
            t_ntt_ptr[l] = t_target_ptr[j * coeff_count + l];
          }
        } else {
          // Perform RNS conversion (modular reduction)
          intel::hexl::EltwiseReduceMod(
              t_ntt_ptr, &t_target_ptr[j * coeff_count], coeff_count,
              moduli[key_index], moduli[key_index], 1);
        }

        // NTT conversion lazy outputs in [0, 4q)
        NTT(n, moduli[key_index]).ComputeForward(t_ntt_ptr, t_ntt_ptr, 4, 4);

        t_operand = t_ntt_ptr;
      }

      // Multiply with keys and modular accumulate products in a lazy fashion
      for (size_t k = 0; k < key_component_count; ++k) {
        // No reduction used; assume intermediate results don't overflow
        for (size_t l = 0; l < coeff_count; ++l) {
          uint64_t t_poly_idx = 2 * (k * coeff_count + l);

          uint64_t mult_op2_idx =
              coeff_count * key_index + k * key_modulus_size * coeff_count + l;

          uint128_t prod =
              MultiplyUInt64(t_operand[l], k_switch_keys[j][mult_op2_idx]);

          // TODO(fboemer): add uint128
          uint128_t low = t_poly_lazy_ptr[t_poly_idx];
          uint128_t hi = t_poly_lazy_ptr[t_poly_idx + 1];
          uint128_t x = (hi << 64) + low;
          uint128_t sum = prod + x;
          uint64_t sum_hi = static_cast<uint64_t>(sum >> 64);
          uint64_t sum_lo = static_cast<uint64_t>(sum);
          t_poly_lazy_ptr[t_poly_idx] = sum_lo;
          t_poly_lazy_ptr[t_poly_idx + 1] = sum_hi;
        }
      }
    }

    // PolyIter pointing to the destination t_poly_prod, shifted to the
    // appropriate modulus
    uint64_t* t_poly_prod_iter_ptr = &t_poly_prod[i * coeff_count];

    // Final modular reduction
    for (size_t k = 0; k < key_component_count; ++k) {
      for (size_t l = 0; l < coeff_count; ++l) {
        uint64_t accumulator_idx = 2 * coeff_count * k + 2 * l;
        uint64_t poly_iter_idx = coeff_count * rns_modulus_size * k + l;

        t_poly_prod_iter_ptr[poly_iter_idx] = BarrettReduce128(
            accumulator_ptr[accumulator_idx + 1],
            accumulator_ptr[accumulator_idx], moduli[key_index]);
      }
    }
  }

  uint64_t* data_array = result;
  for (size_t key_component = 0; key_component < key_component_count;
       ++key_component) {
    uint64_t* t_poly_prod_it =
        &t_poly_prod[key_component * coeff_count * rns_modulus_size];
    uint64_t* t_last = &t_poly_prod_it[decomp_modulus_size * coeff_count];

    NTT(n, moduli[key_modulus_size - 1]).ComputeInverse(t_last, t_last, 2, 2);

    uint64_t qk = moduli[key_modulus_size - 1];
    uint64_t qk_half = qk >> 1;

    for (size_t i = 0; i < coeff_count; ++i) {
      uint64_t barrett_factor =
          MultiplyFactor(1, 64, moduli[key_modulus_size - 1]).BarrettFactor();
      t_last[i] = BarrettReduce64(t_last[i] + qk_half,
                                  moduli[key_modulus_size - 1], barrett_factor);
    }

    for (size_t i = 0; i < decomp_modulus_size; ++i) {
      // (ct mod 4qk) mod qi
      uint64_t qi = moduli[i];

      // TODO(fboemer): Use input_mod_factor != 0 when qk / qi < 4
      // TODO(fboemer): Use output_mod_factor == 4?
      uint64_t input_mod_factor = (qk > qi) ? moduli[i] : 2;
      if (qk > qi) {
        intel::hexl::EltwiseReduceMod(t_ntt_ptr, t_last, coeff_count, moduli[i],
                                      input_mod_factor, 1);
      } else {
        for (size_t coeff_idx = 0; coeff_idx < coeff_count; ++coeff_idx) {
          t_ntt_ptr[coeff_idx] = t_last[coeff_idx];
        }
      }

      // Lazy subtraction, results in [0, 2*qi), since fix is in [0, qi].
      uint64_t barrett_factor =
          MultiplyFactor(1, 64, moduli[i]).BarrettFactor();
      uint64_t fix = qi - BarrettReduce64(qk_half, moduli[i], barrett_factor);
      for (size_t l = 0; l < coeff_count; ++l) {
        t_ntt_ptr[l] += fix;
      }

      uint64_t qi_lazy = qi << 1;  // some multiples of qi

      NTT(n, moduli[i]).ComputeForward(t_ntt_ptr, t_ntt_ptr, 4, 4);
      // Since SEAL uses at most 60bit moduli, 8*qi < 2^63.
      qi_lazy = qi << 2;

      // ((ct mod qi) - (ct mod qk)) mod qi
      uint64_t* t_ith_poly = &t_poly_prod_it[i * coeff_count];
      for (size_t k = 0; k < coeff_count; ++k) {
        t_ith_poly[k] = t_ith_poly[k] + qi_lazy - t_ntt[k];
      }

      // qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
      intel::hexl::EltwiseFMAMod(t_ith_poly, t_ith_poly, modswitch_factors[i],
                                 nullptr, coeff_count, moduli[i], 8);
      uint64_t data_ptr_offset =
          coeff_count * (decomp_modulus_size * key_component + i);

      uint64_t* data_ptr = &data_array[data_ptr_offset];
      intel::hexl::EltwiseAddMod(data_ptr, data_ptr, t_ith_poly, coeff_count,
                                 moduli[i]);
    }
  }
  return;
}

}  // namespace hexl
}  // namespace intel
