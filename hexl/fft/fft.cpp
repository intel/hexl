// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fft.hpp"

#include <iostream>

#include "hexl/logging/logging.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/defines.hpp"

namespace intel {
namespace hexl {

// AllocatorStrategyPtr mallocStrategy = AllocatorStrategyPtr(new
// MallocStrategy);

FFT::FFT(uint64_t degree, double_t* in_scalar,
         std::shared_ptr<AllocatorBase> alloc_ptr)
    : m_degree(degree),
      scalar(in_scalar),
      m_alloc(alloc_ptr),
      m_aligned_alloc(AlignedAllocator<double_t, 64>(m_alloc)),
      m_complex_root_of_unity_powers_real(m_aligned_alloc),
      m_complex_root_of_unity_powers_imag(m_aligned_alloc) {
  // HEXL_CHECK(CheckArguments(degree, q), "");
  // HEXL_CHECK(IsPrimitiveRoot(m_w, 2 * degree, q),
  //           m_w << " is not a primitive 2*" << degree << "'th root of
  //           unity");

  m_degree_bits = Log2(m_degree);
}

bool FFT::CheckArguments(uint64_t degree, uint64_t modulus) {
  HEXL_UNUSED(degree);
  HEXL_UNUSED(modulus);
  HEXL_CHECK(IsPowerOfTwo(degree),
             "degree " << degree << " is not a power of 2");

  HEXL_CHECK(modulus % (2 * degree) == 1,
             "modulus mod 2n != 1");  // IS this needed?
  HEXL_CHECK(IsPrime(modulus), "modulus is not prime");

  return true;
}

void FFT::ComputeComplexPrimitiveRootOfUnityPowers() {}

void FFT::ComputeComplexRootOfUnityPowers() {}

void FFT::ComputeForwardFFT(double_t* result_real, double_t* result_imag,
                            const double_t* operand_real,
                            const double_t* operand_imag,
                            const double_t* roots_real,
                            const double_t* roots_imag) {
  HEXL_CHECK(result_real != nullptr, "result_real == nullptr");
  HEXL_CHECK(result_imag != nullptr, "result_imag == nullptr");
  HEXL_CHECK(operand_real != nullptr, "operand_real == nullptr");
  HEXL_CHECK(operand_imag != nullptr, "operand_imag == nullptr");
  HEXL_CHECK(roots_real != nullptr, "W_real == nullptr");
  HEXL_CHECK(roots_imag != nullptr, "W_imag == nullptr");

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ FwdFFT");

  ComplexFwdTransformToBitReverseAVX512(result_real, result_imag, operand_real,
                                        operand_imag, roots_real, roots_imag,
                                        m_degree, scalar);
  return;
#endif
}

void FFT::ComputeForwardFFTI(double_t* result_interleaved,
                             const double_t* operand_interleaved,
                             const double_t* roots_interleaved) {
  HEXL_CHECK(result_interleaved != nullptr, "result_real == nullptr");
  HEXL_CHECK(operand_interleaved != nullptr, "operand_real == nullptr");
  HEXL_CHECK(roots_interleaved != nullptr, "W_real == nullptr");

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ FwdFFT");

  ComplexFwdTransformToBitReverseAVX512I(result_interleaved,
                                         operand_interleaved, roots_interleaved,
                                         m_degree, scalar);
  return;
#endif
}

void FFT::BuildFloatingPoints(double_t* res, const uint64_t* plain,
                              const uint64_t* threshold,
                              const uint64_t* decryption_modulus,
                              const double_t inv_scale, size_t mod_size,
                              size_t coeff_count) {
#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ FwdFFT");

  BuildFloatingPointsAVX512(res, plain, threshold, decryption_modulus,
                            inv_scale, mod_size, coeff_count);
  return;
#endif
}

}  // namespace hexl
}  // namespace intel
