// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hexl/fft/fft-native.hpp"

#include <cstring>
#include <iostream>

#include "hexl/logging/logging.hpp"
#include "util/util-internal.hpp"

namespace intel {
namespace hexl {

inline void ComplexFwdButterflyRadix2(std::complex<double>* X_r,
                                      std::complex<double>* Y_r,
                                      const std::complex<double>* X_op,
                                      const std::complex<double>* Y_op,
                                      const std::complex<double> W) {
  HEXL_VLOG(5, "ComplexFwdButterflyRadix2");
  HEXL_VLOG(5, "Inputs: X_op " << *X_op << ", Y_op " << *Y_op << ", W " << W);
  std::complex<double> U = *X_op;
  std::complex<double> V = *Y_op * W;
  *X_r = U + V;
  *Y_r = U - V;
  HEXL_VLOG(5, "Output X " << *X_r << ", Y " << *Y_r);
}

inline void ComplexInvButterflyRadix2(std::complex<double>* X_r,
                                      std::complex<double>* Y_r,
                                      const std::complex<double>* X_op,
                                      const std::complex<double>* Y_op,
                                      const std::complex<double> W) {
  HEXL_VLOG(5, "ComplexInvButterflyRadix2");
  HEXL_VLOG(5, "Inputs: X_op " << *X_op << ", Y_op " << *Y_op << ", W " << W);
  std::complex<double> U = *X_op;
  *X_r = U + *Y_op;
  *Y_r = (U - *Y_op) * W;
  HEXL_VLOG(5, "Output X " << *X_r << ", Y " << *Y_r);
}

void Forward_FFT_Radix2(std::complex<double>* result,
                        const std::complex<double>* operand,
                        const std::complex<double>* root_of_unity_powers,
                        const uint64_t n) {
  HEXL_CHECK(IsPowerOfTwo(n), "degree " << n << " is not a power of 2");
  HEXL_CHECK(root_of_unity_powers != nullptr,
             "root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(result != nullptr, "result == nullptr");

  size_t bits = log2(n);
  for (size_t i = 0; i < n; ++i) {
    size_t j = ReverseBits(i, bits);
    if (result == operand) {
      if (j > i) {
        std::complex<double> tmp = operand[i];
        result[i] = operand[j];
        result[j] = tmp;
      }
    } else {
      result[i] = operand[j];
    }
  }

  uint64_t n_div_2 = (n >> 1);
  size_t gap = 1;
  size_t root_index = 0;
  size_t m = n_div_2;

  for (; m > 0; m >>= 1) {
    std::cout << "Fwd S = " << gap << " m = " << m << std::endl;
    size_t j1 = 0;

    switch (gap) {
      case 1: {
        std::cout << "Fwd Loop J1 = " << j1 << " gap = " << gap << " step = 1 "
                  << std::endl;
        const std::complex<double>* W = &root_of_unity_powers[root_index++];
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }

          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = result + j1;
          const std::complex<double>* Y_op = X_op + gap;

          int xc = j1;
          int yc = xc + gap;

          std::cout << "Fwd \t x[" << xc << "] = " << *X_op << " y[" << yc
                    << "] = " << *Y_op << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *W);
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
        }
        break;
      }
      case 2: {
        std::cout << "Fwd Loop J1 = " << j1 << " gap = " << gap << " step = 2 "
                  << std::endl;
        const std::complex<double>* W = &root_of_unity_powers[root_index];
        root_index += 2;
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }

          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;

          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *W);
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 1) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 1));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
        }
        break;
      }
      case 4: {
        std::cout << "Fwd Loop J1 = " << j1 << " gap = " << gap << " step = 4 "
                  << std::endl;
        const std::complex<double>* W = &root_of_unity_powers[root_index];
        root_index += 4;
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }

          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *W);
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 1) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 1));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 2) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 2));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 3) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 3));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
        }
        break;
      }
      case 8: {
        std::cout << "Fwd Loop J1 = " << j1 << " gap = " << gap
                  << " step = " << gap << std::endl;
        const std::complex<double>* W = &root_of_unity_powers[root_index];
        root_index += 8;
        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }

          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                    << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *W);
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 1) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 1));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 2) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 2));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 3) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 3));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 4) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 4));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 5) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 5));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 6) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 6));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
          X_r++;
          Y_r++;
          X_op++;
          Y_op++;
          xc++;
          yc++;
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << " Wg[" << root_index
                    << "] = " << *(W + 7) << std::endl;
          ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 7));
          std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                    << "] = " << *Y_r << std::endl;
        }
        break;
      }
      default: {
        std::cout << "Fwd Default Loop J1 = " << j1 << " gap = " << gap
                  << " step = " << gap << std::endl;

        for (size_t i = 0; i < m; i++) {
          if (i != 0) {
            j1 += (gap << 1);
          }
          const std::complex<double>* W = &root_of_unity_powers[root_index];
          std::complex<double>* X_r = result + j1;
          std::complex<double>* Y_r = X_r + gap;
          const std::complex<double>* X_op = X_r;
          const std::complex<double>* Y_op = Y_r;
          int xc = j1;
          int yc = xc + gap;
          HEXL_LOOP_UNROLL_8
          for (size_t j = 0; j < gap; j += 8) {
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << " Wg[" << root_index << "] = " << *W
                      << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *W);
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << std::endl;
            X_r++;
            Y_r++;
            X_op++;
            Y_op++;
            xc++;
            yc++;
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << " Wg[" << root_index
                      << "] = " << *(W + 1) << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 1));
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << std::endl;
            X_r++;
            Y_r++;
            X_op++;
            Y_op++;
            xc++;
            yc++;
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << " Wg[" << root_index
                      << "] = " << *(W + 2) << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 2));
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << std::endl;
            X_r++;
            Y_r++;
            X_op++;
            Y_op++;
            xc++;
            yc++;
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << " Wg[" << root_index
                      << "] = " << *(W + 3) << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 3));
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << std::endl;
            X_r++;
            Y_r++;
            X_op++;
            Y_op++;
            xc++;
            yc++;
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << " Wg[" << root_index
                      << "] = " << *(W + 4) << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 4));
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << std::endl;
            X_r++;
            Y_r++;
            X_op++;
            Y_op++;
            xc++;
            yc++;
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << " Wg[" << root_index
                      << "] = " << *(W + 5) << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 5));
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << std::endl;
            X_r++;
            Y_r++;
            X_op++;
            Y_op++;
            xc++;
            yc++;
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << " Wg[" << root_index
                      << "] = " << *(W + 6) << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 6));
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << std::endl;
            X_r++;
            Y_r++;
            X_op++;
            Y_op++;
            xc++;
            yc++;
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << " Wg[" << root_index
                      << "] = " << *(W + 7) << std::endl;
            ComplexFwdButterflyRadix2(X_r, Y_r, X_op, Y_op, *(W + 7));
            std::cout << "Fwd \t x[" << xc << "] = " << *X_r << " y[" << yc
                      << "] = " << *Y_r << std::endl;
            X_r++;
            Y_r++;
            X_op++;
            Y_op++;
            xc++;
            yc++;

            W += 8;
          }
        }
        root_index += gap;
      }
    }
    gap <<= 1;
  }

  // When M is too short it only needs the final stage butterfly. Copying here
  // in the case of out-of-place.
  if (result != operand && n == 2) {
    std::memcpy(result, operand, n * sizeof(std::complex<double>));
  }
}

void Inverse_FFT_Radix2(std::complex<double>* result,
                        const std::complex<double>* operand,
                        const std::complex<double>* inv_root_of_unity_powers,
                        const uint64_t n) {
  HEXL_CHECK(IsPowerOfTwo(n), "degree " << n << " is not a power of 2");
  HEXL_CHECK(inv_root_of_unity_powers != nullptr,
             "inv_root_of_unity_powers == nullptr");
  HEXL_CHECK(operand != nullptr, "operand == nullptr");
  HEXL_CHECK(result != nullptr, "result == nullptr");

  Forward_FFT_Radix2(result, operand, inv_root_of_unity_powers, n);

  for (size_t i = 0; i < n; i++) {
    result[i] /= static_cast<double>(n);
  }
}

}  // namespace hexl
}  // namespace intel
