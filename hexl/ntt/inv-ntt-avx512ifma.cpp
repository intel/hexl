// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <immintrin.h>

#include <functional>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "ntt/inv-ntt-avx512.tpp"
#include "ntt/ntt-avx512-util.hpp"
#include "ntt/ntt-internal.hpp"
#include "util/avx512-util.hpp"
#include "util/avx512ifma-util.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512IFMA
template void InverseTransformFromBitReverseAVX512<NTT::s_ifma_shift_bits>(
    uint64_t* operand, uint64_t degree, uint64_t modulus,
    const uint64_t* inv_root_of_unity_powers,
    const uint64_t* precon_inv_root_of_unity_powers, uint64_t input_mod_factor,
    uint64_t output_mod_factor, uint64_t recursion_depth,
    uint64_t recursion_half);
#endif

}  // namespace hexl
}  // namespace intel
