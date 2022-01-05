// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "hexl/fft/fft.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/util/defines.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_DEBUG
/*TEST(NTT, bad_input) {
  uint64_t N = 8;
  uint64_t modulus = 769;
  std::vector<uint64_t> input;
  std::vector<uint64_t> p_input;
  std::vector<uint64_t> p_times_2_input;
  std::vector<uint64_t> p_times_4_input;

  NTT ntt(N, modulus);

  auto init_inputs = [&]() {
    input = {1, 2, 3, 4, 5, 6, 7, 8};
    p_input = std::vector<uint64_t>(N, modulus);
    p_times_2_input = std::vector<uint64_t>(N, 2 * modulus);
    p_times_4_input = std::vector<uint64_t>(N, 4 * modulus);
  };

  // Forward transform
  // Bad input
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeForward(input.data(), nullptr, 1, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeForward(nullptr, input.data(), 1, 1));
  init_inputs();
  EXPECT_NO_THROW(ntt.ComputeForward(input.data(), input.data(), 1, 1));
  init_inputs();
  EXPECT_NO_THROW(ntt.ComputeForward(p_input.data(), p_input.data(), 4, 4));
  init_inputs();
  EXPECT_ANY_THROW(
      ntt.ComputeForward(p_times_2_input.data(), p_times_2_input.data(), 2, 1));
  init_inputs();
  EXPECT_NO_THROW(
      ntt.ComputeForward(p_times_2_input.data(), p_times_2_input.data(), 4, 4));
  init_inputs();
  EXPECT_ANY_THROW(
      ntt.ComputeForward(p_times_4_input.data(), p_times_4_input.data(), 4, 4));
  init_inputs();

  // Bad mod factors
  EXPECT_NO_THROW(ntt.ComputeForward(input.data(), input.data(), 2, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeForward(input.data(), input.data(), 123, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeForward(input.data(), input.data(), 2, 123));
  init_inputs();

  // Inverse transform

  // Bad input
  EXPECT_ANY_THROW(ntt.ComputeInverse(input.data(), nullptr, 1, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeInverse(nullptr, input.data(), 1, 1));
  init_inputs();

  EXPECT_NO_THROW(ntt.ComputeInverse(input.data(), input.data(), 1, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeInverse(p_input.data(), p_input.data(), 1, 1));
  init_inputs();
  EXPECT_NO_THROW(ntt.ComputeInverse(p_input.data(), p_input.data(), 2, 2));
  init_inputs();
  EXPECT_ANY_THROW(
      ntt.ComputeInverse(p_times_2_input.data(), p_times_2_input.data(), 2, 2));
  init_inputs();

  // Bad mod factors
  EXPECT_NO_THROW(ntt.ComputeInverse(input.data(), input.data(), 1, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeInverse(input.data(), input.data(), 123, 1));
  init_inputs();
  EXPECT_ANY_THROW(ntt.ComputeInverse(input.data(), input.data(), 1, 123));
  init_inputs();
}*/
#endif

TEST(FFT, Powers) {
  {
    std::vector<double_t> result_real{0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 1, 2, 3, 4, 5, 6};
    std::vector<double_t> result_imag{0, 1, 2, 3, 4, 5, 6, 7,
                                      8, 9, 1, 2, 3, 4, 5, 6};
    std::vector<double_t> operand_real{0, 1, 2, 3, 4, 5, 6, 7,
                                       8, 9, 1, 2, 3, 4, 5, 6};
    std::vector<double_t> operand_imag{0, 1, 2, 3, 4, 5, 6, 7,
                                       8, 9, 1, 2, 3, 4, 5, 6};
    std::vector<double_t> roots_real{0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 1, 2, 3, 4, 5, 6};
    std::vector<double_t> roots_imag{0, 1, 2, 3, 4, 5, 6, 7,
                                     8, 9, 1, 2, 3, 4, 5, 6};
    uint64_t degree = 16;
    double_t in_scalar = 4;
    FFT fft(degree, &in_scalar);
    fft.ComputeForwardFFTRI(result_real.data(), result_imag.data(),
                            operand_real.data(), operand_imag.data(),
                            roots_real.data(), roots_imag.data());

    AssertEqual(result_real, operand_real);
    HEXL_VLOG(1, "result_real[5] " << result_real[5]);
    // ASSERT_EQ(1ULL, 1);
  }
}

}  // namespace hexl
}  // namespace intel
