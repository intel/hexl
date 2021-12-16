// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fft.hpp"

#include "hexl/fft/fwd-fft-avx512.hpp"
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

void FFT::ComputeComplexPrimitiveRootOfUnityPowers() {
  /*uint64_t m = coeff_degree << 1;

  // Generate 1/8 of all roots.
  for (size_t i = 0; i <= m / 8; i++) {
    complex_root_of_unity_powers_phase[i] =
        2 * PI_ * static_cast<double>(i) / static_cast<double>(m);
  }*/
}

void FFT::ComputeComplexRootOfUnityPowers() {
  /*uint64_t m = coeff_degree << 1;
  uint64_t m_bits = Log2(m);

  AlignedVector64<uint64_t> complex_root_of_unity_powers_phase;

  // Powers of the primitive 2n-th root have 4-fold symmetry
  if (m >= 8) {
    complex_roots_ = make_shared<util::ComplexRoots>(
        util::ComplexRoots(static_cast<size_t>(m), pool_));
    for (size_t i = 1; i < coeff_degree; i++) {
      m_complex_root_of_unity_powers_phase[i] =
          complex_roots_->get_root(ReverseBits(i, m_bits));
      m_inv_complex_root_of_unity_powers_phase[i] =
          conj(complex_roots_->get_root(ReverseBits(i - 1, m_bits) + 1));
    }
  } else if (m == 4) {
    m_complex_root_of_unity_powers_phase[1] = 0;
    m_inv_complex_root_of_unity_powers_phase[1] = 0;
    m_complex_root_of_unity_powers_phase[2] = 1;
    m_inv_complex_root_of_unity_powers_phase[2] = -1;
  }*/
}

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

}  // namespace hexl
}  // namespace intel
