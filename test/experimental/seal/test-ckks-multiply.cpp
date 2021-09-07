// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "hexl/experimental/seal/ckks-multiply.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

TEST(CkksMultiply, small_one_mod) {
  size_t coeff_count = 3;
  std::vector<uint64_t> moduli{10};

  std::vector<uint64_t> op1{1, 2, 3,  //
                            4, 5, 6};
  std::vector<uint64_t> op2{2, 4, 6,  //
                            8, 1, 3};
  std::vector<uint64_t> out(3 * coeff_count * moduli.size(), 0);

  std::vector<uint64_t> exp_out{
      (1 * 2 % 10),         (2 * 4 % 10),         (3 * 6 % 10),          //
      (1 * 8 + 4 * 2) % 10, (2 * 1 + 5 * 4) % 10, (3 * 3 + 6 * 6) % 10,  //
      (4 * 8 % 10),         (5 * 1 % 10),         (6 * 3 % 10)           //
  };

  CkksMultiply(out.data(), op1.data(), op2.data(), coeff_count, moduli.data(),
               moduli.size());

  CheckEqual(out, exp_out);
}

TEST(CkksMultiply, small_one_mod_inplace) {
  size_t coeff_count = 3;
  std::vector<uint64_t> moduli{10};

  std::vector<uint64_t> op1{
      1, 2, 3,  // poly 1
      4, 5, 6,  // poly 2
      0, 0, 0   // poly 3 (output)
  };
  std::vector<uint64_t> op2{2, 4, 6,  //
                            8, 1, 3};

  std::vector<uint64_t> exp_out{
      (1 * 2 % 10),         (2 * 4 % 10),         (3 * 6 % 10),          //
      (1 * 8 + 4 * 2) % 10, (2 * 1 + 5 * 4) % 10, (3 * 3 + 6 * 6) % 10,  //
      (4 * 8 % 10),         (5 * 1 % 10),         (6 * 3 % 10)           //
  };

  CkksMultiply(op1.data(), op1.data(), op2.data(), coeff_count, moduli.data(),
               moduli.size());

  CheckEqual(op1, exp_out);
}

TEST(CkksMultiply, small_two_mod) {
  size_t coeff_count = 3;
  std::vector<uint64_t> moduli{10, 20};

  std::vector<uint64_t> op1{
      1,  2,  3,   // poly 1 mod 10
      11, 12, 13,  // poly 1 mod 20
      4,  5,  6,   // poly 2 mod 10
      14, 15, 16   // poly 2 mod 20
  };
  std::vector<uint64_t> op2{
      2,  4,  6,   // poly 1 mod 10
      12, 14, 16,  // poly 1 mod 20
      8,  1,  3,   // poly 2 mod 10
      18, 11, 13   // poly 2 mod 20
  };
  std::vector<uint64_t> out(3 * coeff_count * moduli.size(), 0);

  std::vector<uint64_t> exp_out{(1 * 2 % 10),  // poly 1
                                (2 * 4 % 10),
                                (3 * 6 % 10),
                                (11 * 12 % 20),
                                (12 * 14 % 20),
                                (13 * 16 % 20),
                                (1 * 8 + 4 * 2) % 10,  // poly 2
                                (2 * 1 + 5 * 4) % 10,
                                (3 * 3 + 6 * 6) % 10,
                                (11 * 18 + 14 * 12) % 20,
                                (12 * 11 + 15 * 14) % 20,
                                (13 * 13 + 16 * 16) % 20,
                                (4 * 8 % 10),  // poly 3
                                (5 * 1 % 10),
                                (6 * 3 % 10),
                                (14 * 18 % 20),
                                (15 * 11 % 20),
                                (16 * 13 % 20)};

  CkksMultiply(out.data(), op1.data(), op2.data(), coeff_count, moduli.data(),
               moduli.size());

  CheckEqual(out, exp_out);
}

}  // namespace hexl
}  // namespace intel
