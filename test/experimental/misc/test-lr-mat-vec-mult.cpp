// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "hexl/experimental/misc/lr-mat-vec-mult.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

TEST(LinRegMatrixVectorMultiply, small_one_mod) {
  size_t num_weights = 2;
  size_t coeff_count = 3;
  std::vector<uint64_t> moduli{10};
  std::vector<uint64_t> op1{1, 1, 1,           // w0
                            4, 5, 6, 2, 2, 2,  // w1
                            4, 5, 6};
  //        t0  t1  t2
  std::vector<uint64_t> op2{3, 4, 5,           // c0 = { t00 t10 t20 }
                            8, 1, 3, 1, 2, 3,  // c1 = { t01 t11 t21 }
                            8, 1, 3};

  // w0 .* c0 + w1 .* c1

  std::vector<uint64_t> out(num_weights * moduli.size() * 3 * coeff_count, 0);

  std::vector<uint64_t> exp_out{
      ((1 * 3 % 10) + (2 * 1 % 10)) % 10,
      ((1 * 4 % 10) + (2 * 2 % 10)) % 10,
      ((1 * 5 % 10) + (2 * 3 % 10)) % 10,  //
      (((1 * 8 + 4 * 3) % 10) + ((2 * 8 + 4 * 1) % 10)) % 10,
      (((1 * 1 + 5 * 4) % 10 + (2 * 1 + 5 * 2) % 10)) % 10,
      (((1 * 3 + 6 * 5) % 10) + ((2 * 3 + 6 * 3) % 10)) % 10,  //
      ((4 * 8 % 10) + (4 * 8 % 10)) % 10,
      ((5 * 1 % 10) + (5 * 1 % 10)) % 10,
      ((6 * 3 % 10) + (6 * 3 % 10)) % 10,  //

      (2 * 1 % 10),
      (2 * 2 % 10),
      (2 * 3 % 10),  //
      (2 * 8 + 4 * 1) % 10,
      (2 * 1 + 5 * 2) % 10,
      (2 * 3 + 6 * 3) % 10,  //
      (4 * 8 % 10),
      (5 * 1 % 10),
      (6 * 3 % 10)  //
  };

  LinRegMatrixVectorMultiply(out.data(), op1.data(), op2.data(), coeff_count,
                             moduli.data(), moduli.size(), num_weights);

  CheckEqual(out, exp_out);
}

}  // namespace hexl
}  // namespace intel
