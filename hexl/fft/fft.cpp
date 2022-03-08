// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fft.hpp"

#include "hexl/fft/fft-native.hpp"
#include "hexl/logging/logging.hpp"

namespace intel {
namespace hexl {

FFT::FFT(uint64_t degree, std::shared_ptr<AllocatorBase> alloc_ptr)
    : m_degree(degree),
      m_alloc(alloc_ptr),
      m_aligned_alloc(AlignedAllocator<double, 64>(m_alloc)),
      m_complex_roots_of_unity(m_aligned_alloc) {
  HEXL_CHECK(IsPowerOfTwo(degree),
             "degree " << degree << " is not a power of 2");
  HEXL_CHECK(degree > 8, "degree should be bigger than 8");

  m_degree_bits = Log2(m_degree);
  ComputeComplexRootsOfUnity();
}

inline std::complex<double> swap_real_imag(std::complex<double> c) {
  return std::complex<double>(c.imag(), c.real());
}

void FFT::ComputeComplexRootsOfUnity() {
  AlignedVector64<std::complex<double>> roots(m_degree, 0, m_aligned_alloc);
  AlignedVector64<std::complex<double>> inv_roots(m_degree, 0, m_aligned_alloc);
  AlignedVector64<double> roots_d(2 * m_degree, 0, m_aligned_alloc);

  // PI value used to calculate the roots of unity
  static constexpr double PI_ = 3.1415926535897932384626433832795028842;

  const std::complex<double> J(0, 1);
  size_t gap = 1;
  size_t root_index = 1;
  std::cout << "m_degree " << m_degree << std::endl;
  for (size_t m = m_degree >> 1; m > 0; m >>= 1) {
    std::complex<double> w(1, 0);
    std::complex<double> wm =
        std::exp(J * ((-1 * PI_) / static_cast<double>(gap)));
    for (size_t j = 0; j < gap; ++j, root_index++) {
      size_t idx_r = (root_index & 7) + (root_index >> 3) * 16;
      size_t idx_i = idx_r + 8;
      roots_d[idx_r] = w.real();
      roots_d[idx_i] = w.imag();

      roots[root_index] = w;
      inv_roots[root_index] = std::conj(w);
      std::cout << "i " << root_index << " W " << roots[root_index] << " => "
                << idx_r << " = " << roots_d[idx_r] << " , " << idx_i << " = "
                << roots_d[idx_i] << std::endl;
      w *= wm;
    }
    gap <<= 1;
  }

  /*
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
    roots[i] = roots_of_unity[ReverseBits(i, m_degree_bits)];
    inv_roots[i] =
        std::conj(roots_of_unity[ReverseBits(i - 1, m_degree_bits) + 1]);
  }
  m_complex_roots_of_unity = roots_of_unity;
  */

  m_complex_roots_of_unity = roots;
  m_complex_roots_of_unity_d = roots_d;
  m_inv_complex_roots_of_unity = inv_roots;
}

void FFT::ComputeForwardFFT(std::complex<double>* result,
                            const std::complex<double>* operand) {
  HEXL_CHECK(result != nullptr, "result == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ FwdFFT");

  Forward_FFT_AVX512(
      &(reinterpret_cast<double(&)[2]>(result[0]))[0],
      &(reinterpret_cast<const double(&)[2]>(operand[0]))[0],
      &(reinterpret_cast<const double(&)[2]>(m_complex_roots_of_unity[0]))[0],
      m_degree);
  return;
#else
  HEXL_VLOG(3, "Calling Native FwdFFT");
  Forward_FFT_Radix2(result, operand, m_complex_roots_of_unity.data(),
                     m_degree);
  return;
#endif
}

void FFT::ComputeInverseFFT(std::complex<double>* result,
                            const std::complex<double>* operand) {
  HEXL_CHECK(result != nullptr, "result==nullptr");
  HEXL_CHECK(operand != nullptr, "operand==nullptr");

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ InvFFT");

  Inverse_FFT_AVX512(&(reinterpret_cast<double(&)[2]>(result[0]))[0],
                     &(reinterpret_cast<const double(&)[2]>(operand[0]))[0],
                     &(reinterpret_cast<const double(&)[2]>(
                         m_inv_complex_roots_of_unity[0]))[0],
                     m_degree);

  return;
#else
  HEXL_VLOG(3, "Calling Native InvFFT");
  Inverse_FFT_Radix2(result, operand, m_inv_complex_roots_of_unity.data(),
                     m_degree);
  return;
#endif
}

}  // namespace hexl
}  // namespace intel
