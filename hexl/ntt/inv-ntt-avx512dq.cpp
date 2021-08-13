// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include <functional>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "ntt/inv-ntt-avx512.tpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
template void InverseTransformFromBitReverseAVX512<32>(
    uint64_t* operand, uint64_t degree, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);

template void InverseTransformFromBitReverseAVX512<NTT::s_default_shift_bits>(
    uint64_t* operand, uint64_t degree, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);
#endif

}  // namespace hexl
}  // namespace intel
