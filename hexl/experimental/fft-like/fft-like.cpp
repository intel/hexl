// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/experimental/fft-like/fft-like.hpp"

#include "hexl/experimental/fft-like/fft-like-native.hpp"
#include "hexl/logging/logging.hpp"

namespace intel {
namespace hexl {

FFTLike::FFTLike(uint64_t degree, double* in_scalar,
                 std::shared_ptr<AllocatorBase> alloc_ptr)
    : m_degree(degree),
      scalar(in_scalar),
      m_alloc(alloc_ptr),
      m_aligned_alloc(AlignedAllocator<double, 64>(m_alloc)),
      m_complex_roots_of_unity(m_aligned_alloc) {
  HEXL_CHECK(IsPowerOfTwo(degree),
             "degree " << degree << " is not a power of 2");
  HEXL_CHECK(degree > 8, "degree should be bigger than 8");

  m_degree_bits = Log2(m_degree);
  ComputeComplexRootsOfUnity();

  if (scalar != nullptr) {
    scale = *scalar / static_cast<double>(degree);
    inv_scale = 1.0 / *scalar;
  }
}

inline std::complex<double> swap_real_imag(std::complex<double> c) {
  return std::complex<double>(c.imag(), c.real());
}

void FFTLike::ComputeComplexRootsOfUnity() {
  AlignedVector64<std::complex<double>> roots_of_unity(m_degree, 0,
                                                       m_aligned_alloc);
  AlignedVector64<std::complex<double>> roots_in_bit_reverse(m_degree, 0,
                                                             m_aligned_alloc);
  AlignedVector64<std::complex<double>> inv_roots_in_bit_reverse(
      m_degree, 0, m_aligned_alloc);
  uint64_t roots_degree = static_cast<uint64_t>(m_degree) << 1;  // degree > 2

  // PI value used to calculate the roots of unity
  static constexpr double PI_ = 3.1415926535897932384626433832795028842;

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

void FFTLike::ComputeForwardFFTLike(std::complex<double>* result,
                                    const std::complex<double>* operand,
                                    const double* in_scale) {
  HEXL_CHECK(result != nullptr, "result == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");

  const double* out_scale = nullptr;
  if (scalar != nullptr) {
    out_scale = &inv_scale;
  } else if (in_scale != nullptr) {
    out_scale = in_scale;
  }

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ FwdFFTLike");

  Forward_FFTLike_ToBitReverseAVX512(
      &(reinterpret_cast<double(&)[2]>(result[0]))[0],
      &(reinterpret_cast<const double(&)[2]>(operand[0]))[0],
      &(reinterpret_cast<const double(&)[2]>(m_complex_roots_of_unity[0]))[0],
      m_degree, out_scale);
  return;
#else
  HEXL_VLOG(3, "Calling Native FwdFFTLike");
  Forward_FFTLike_ToBitReverseRadix2(
      result, operand, m_complex_roots_of_unity.data(), m_degree, out_scale);
  return;
#endif
}

void FFTLike::ComputeInverseFFTLike(std::complex<double>* result,
                                    const std::complex<double>* operand,
                                    const double* in_scale) {
  HEXL_CHECK(result != nullptr, "result==nullptr");
  HEXL_CHECK(operand != nullptr, "operand==nullptr");

  const double* out_scale = nullptr;
  if (scalar != nullptr) {
    out_scale = &scale;
  } else if (in_scale != nullptr) {
    out_scale = in_scale;
  }

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ InvFFTLike");

  Inverse_FFTLike_FromBitReverseAVX512(
      &(reinterpret_cast<double(&)[2]>(result[0]))[0],
      &(reinterpret_cast<const double(&)[2]>(operand[0]))[0],
      &(reinterpret_cast<const double(&)[2]>(
          m_inv_complex_roots_of_unity[0]))[0],
      m_degree, out_scale);

  return;
#else
  HEXL_VLOG(3, "Calling Native InvFFTLike");
  Inverse_FFTLike_FromBitReverseRadix2(result, operand,
                                       m_inv_complex_roots_of_unity.data(),
                                       m_degree, out_scale);
  return;
#endif
}

void FFTLike::BuildFloatingPoints(std::complex<double>* res,
                                  const uint64_t* plain,
                                  const uint64_t* threshold,
                                  const uint64_t* decryption_modulus,
                                  const double in_inv_scale, size_t mod_size,
                                  size_t coeff_count) {
  HEXL_UNUSED(res);
  HEXL_UNUSED(plain);
  HEXL_UNUSED(threshold);
  HEXL_UNUSED(decryption_modulus);
  HEXL_UNUSED(in_inv_scale);
  HEXL_UNUSED(mod_size);
  HEXL_UNUSED(coeff_count);

#ifdef HEXL_HAS_AVX512DQ
  HEXL_VLOG(3, "Calling 64-bit AVX512-DQ BuildFloatingPoints");

  BuildFloatingPointsAVX512(&(reinterpret_cast<double(&)[2]>(res[0]))[0], plain,
                            threshold, decryption_modulus, in_inv_scale,
                            mod_size, coeff_count);
  return;
#endif
}

}  // namespace hexl
}  // namespace intel
