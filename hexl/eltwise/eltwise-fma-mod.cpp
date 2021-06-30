// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/eltwise/eltwise-fma-mod.hpp"

#include <algorithm>

#include "eltwise/eltwise-fma-mod-avx512.hpp"
#include "eltwise/eltwise-fma-mod-internal.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

void EltwiseFMAMod(uint64_t* result, const uint64_t* arg1, uint64_t arg2,
                   const uint64_t* arg3, uint64_t n, uint64_t modulus,
                   uint64_t input_mod_factor) {
  HEXL_CHECK(result != nullptr, "Require result != nullptr");
  HEXL_CHECK(arg1 != nullptr, "Require arg1 != nullptr");
  HEXL_CHECK(n != 0, "Require n != 0")
  HEXL_CHECK(modulus > 1, "Require modulus > 1");
  HEXL_CHECK(modulus < (1ULL << 61), "Require modulus < (1ULL << 61)");
  HEXL_CHECK(
      input_mod_factor == 1 || input_mod_factor == 2 || input_mod_factor == 4 ||
          input_mod_factor == 8,
      "input_mod_factor must be 1, 2, 4, or 8. Got " << input_mod_factor);
  HEXL_CHECK(
      arg2 < input_mod_factor * modulus,
      "arg2 " << arg2 << " exceeds bound " << (input_mod_factor * modulus));

  HEXL_CHECK_BOUNDS(arg1, n, input_mod_factor * modulus,
                    "arg1 value " << (*std::max_element(arg1, arg1 + n))
                                  << " in EltwiseFMAMod exceeds bound "
                                  << (input_mod_factor * modulus));
  HEXL_CHECK(arg3 == nullptr || (*std::max_element(arg3, arg3 + n) <
                                 (input_mod_factor * modulus)),
             "arg3 value in EltwiseFMAMod exceeds bound "
                 << (input_mod_factor * modulus));

#ifdef HEXL_HAS_AVX512IFMA
  if (has_avx512ifma && input_mod_factor * modulus < (1ULL << 52)) {
    HEXL_VLOG(3, "Calling 52-bit EltwiseFMAModAVX512");

    switch (input_mod_factor) {
      case 1:
        EltwiseFMAModAVX512<52, 1>(result, arg1, arg2, arg3, n, modulus);
        break;
      case 2:
        EltwiseFMAModAVX512<52, 2>(result, arg1, arg2, arg3, n, modulus);
        break;
      case 4:
        EltwiseFMAModAVX512<52, 4>(result, arg1, arg2, arg3, n, modulus);
        break;
      case 8:
        EltwiseFMAModAVX512<52, 8>(result, arg1, arg2, arg3, n, modulus);
        break;
    }
    return;
  }
#endif

#ifdef HEXL_HAS_AVX512DQ
  if (has_avx512dq) {
    HEXL_VLOG(3, "Calling 64-bit EltwiseFMAModAVX512");

    switch (input_mod_factor) {
      case 1:
        EltwiseFMAModAVX512<64, 1>(result, arg1, arg2, arg3, n, modulus);
        break;
      case 2:
        EltwiseFMAModAVX512<64, 2>(result, arg1, arg2, arg3, n, modulus);
        break;
      case 4:
        EltwiseFMAModAVX512<64, 4>(result, arg1, arg2, arg3, n, modulus);
        break;
      case 8:
        EltwiseFMAModAVX512<64, 8>(result, arg1, arg2, arg3, n, modulus);
        break;
    }
    return;
  }
#endif

  HEXL_VLOG(3, "Calling EltwiseFMAModNative");
  switch (input_mod_factor) {
    case 1:
      EltwiseFMAModNative<1>(result, arg1, arg2, arg3, n, modulus);
      break;
    case 2:
      EltwiseFMAModNative<2>(result, arg1, arg2, arg3, n, modulus);
      break;
    case 4:
      EltwiseFMAModNative<4>(result, arg1, arg2, arg3, n, modulus);
      break;
    case 8:
      EltwiseFMAModNative<8>(result, arg1, arg2, arg3, n, modulus);
      break;
  }
}

}  // namespace hexl
}  // namespace intel
