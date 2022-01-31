// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fft.hpp"

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
      m_complex_roots_of_unity(m_aligned_alloc) {
  HEXL_CHECK(IsPowerOfTwo(degree),
             "degree " << degree << " is not a power of 2");
  HEXL_CHECK(degree > 8, "degree should be bigger than 8");

  m_degree_bits = Log2(m_degree);
  ComputeComplexRootsOfUnity();

  if (scalar != nullptr) {
    scale = *scalar / static_cast<double_t>(degree);
    inv_scale = static_cast<double_t>(1.0) / *scalar;
  }
}

inline std::complex<double_t> swap_real_imag(std::complex<double_t> c) {
  return std::complex<double_t>(c.imag(), c.real());
}

void FFT::ComputeComplexRootsOfUnity() {
  AlignedVector64<std::complex<double_t>> roots_of_unity(m_degree, 0,
                                                         m_aligned_alloc);
  AlignedVector64<std::complex<double_t>> roots_in_bit_reverse(m_degree, 0,
                                                               m_aligned_alloc);
  AlignedVector64<std::complex<double_t>> inv_roots_in_bit_reverse(
      m_degree, 0, m_aligned_alloc);
  uint64_t roots_degree = static_cast<uint64_t>(m_degree) << 1;  // degree > 2

  // Generate 1/8 of all roots first.
  size_t i = 0;
  for (; i <= roots_degree / 8; i++) {
    roots_of_unity[i] =
        std::polar<double>(1.0, 2 * PI_ * static_cast<double>(i) /
                                    static_cast<double>(roots_degree));
  }
  // Complete first 4th
  for (; i <= roots_degree / 4; i++) {
    roots_of_unity[i] = swap_real_imag(roots_of_unity[roots_degree / 4 - i]);
  }
  // Get second 4th
  for (; i < roots_degree / 2; i++) {
    roots_of_unity[i] = -std::conj(roots_of_unity[roots_degree / 2 - i]);
  }
  // Put in bit reverse and get inv roots
  for (i = 1; i < m_degree; i++) {
    roots_in_bit_reverse[i] = roots_of_unity[ReverseBits(i, m_degree_bits)];
    inv_roots_in_bit_reverse[i] =
        std::conj(roots_of_unity[ReverseBits(i - 1, m_degree_bits) + 1]);
  }
  m_complex_roots_of_unity = roots_in_bit_reverse;
  m_inv_complex_roots_of_unity = inv_roots_in_bit_reverse;
}

void FFT::ComputeForwardFFT(std::complex<double_t>* result,
                            const std::complex<double_t>* operand,
                            const double_t* in_scale) {
  HEXL_CHECK(result != nullptr, "result == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");

  const double_t* out_scale = nullptr;
  if (scalar != nullptr) {
    out_scale = &inv_scale;
  } else if (in_scale != nullptr) {
    out_scale = in_scale;
  }

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ FwdFFT");

  Forward_FFT_ToBitReverseAVX512(
      &(reinterpret_cast<double_t(&)[2]>(result[0]))[0],
      &(reinterpret_cast<const double_t(&)[2]>(operand[0]))[0],
      &(reinterpret_cast<const double_t(&)[2]>(m_complex_roots_of_unity[0]))[0],
      m_degree, out_scale);
  return;
#else
  HEXL_VLOG(3, "Calling Native FwdFFT");
  Forward_FFT_ToBitReverseRadix2(result, operand, m_complex_roots_of_unity,
                                 m_degree, out_scale);
  return;
#endif
}

void FFT::ComputeInverseFFT(std::complex<double_t>* result,
                            const std::complex<double_t>* operand,
                            const double_t* in_scale) {
  HEXL_CHECK(result != nullptr, "result==nullptr");
  HEXL_CHECK(operand != nullptr, "operand==nullptr");

  const double_t* out_scale = nullptr;
  if (scalar != nullptr) {
    out_scale = &scale;
  } else if (in_scale != nullptr) {
    out_scale = in_scale;
  }

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ InvFFT");

  Inverse_FFT_FromBitReverseAVX512(
      &(reinterpret_cast<double_t(&)[2]>(result[0]))[0],
      &(reinterpret_cast<const double_t(&)[2]>(operand[0]))[0],
      &(reinterpret_cast<const double_t(&)[2]>(
          m_inv_complex_roots_of_unity[0]))[0],
      m_degree, out_scale);

  return;
#else
  HEXL_VLOG(3, "Calling Native InvFFT");
  Inverse_FFT_FromBitReverseRadix2(
      result, operand, m_inv_complex_roots_of_unity, m_degree, out_scale);
  return;
#endif
}

void FFT::BuildFloatingPoints(std::complex<double_t>* res,
                              const uint64_t* plain, const uint64_t* threshold,
                              const uint64_t* decryption_modulus,
                              const double_t in_inv_scale, size_t mod_size,
                              size_t coeff_count) {
#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ BuildFloatingPoints");

  BuildFloatingPointsAVX512(&(reinterpret_cast<double_t(&)[2]>(res[0]))[0],
                            plain, threshold, decryption_modulus, in_inv_scale,
                            mod_size, coeff_count);
  return;
#endif
}

}  // namespace hexl
}  // namespace intel
