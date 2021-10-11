// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "eltwise/eltwise-reduce-mod-internal.hpp"
#include "hexl/eltwise/eltwise-reduce-mod.hpp"
#include "hexl/logging/logging.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "test-util.hpp"

namespace intel {
namespace hexl {

TEST(EltwiseReduceMod, 2_2) {
  std::vector<uint64_t> op{0, 450, 735, 900, 1350, 1459};
  std::vector<uint64_t> exp_out{0, 450, 735, 900, 1350, 1459};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0};

  const uint64_t modulus = 750;
  const uint64_t input_mod_factor = 2;
  const uint64_t output_mod_factor = 2;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, 4_1) {
  std::vector<uint64_t> op{2, 4, 1600, 2500};
  std::vector<uint64_t> exp_out{2, 4, 100, 250};
  std::vector<uint64_t> result{0, 0, 0, 0};

  const uint64_t modulus = 750;
  const uint64_t input_mod_factor = 4;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, 0_1) {
  std::vector<uint64_t> op{2, 4, 1600, 2500};
  std::vector<uint64_t> exp_out{2, 4, 100, 250};
  std::vector<uint64_t> result{0, 0, 0, 0};

  const uint64_t modulus = 750;
  const uint64_t input_mod_factor = modulus;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, 2_1) {
  std::vector<uint64_t> op{0, 450, 735, 900, 1350, 1459};
  std::vector<uint64_t> exp_out{0, 450, 5, 170, 620, 729};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0};

  const uint64_t modulus = 730;
  const uint64_t input_mod_factor = 2;
  const uint64_t output_mod_factor = 1;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

TEST(EltwiseReduceMod, 4_2) {
  std::vector<uint64_t> op{1, 730, 1000, 1460, 2100, 2919};
  std::vector<uint64_t> exp_out{1, 730, 1000, 0, 640, 1459};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0};

  const uint64_t modulus = 730;
  const uint64_t input_mod_factor = 4;
  const uint64_t output_mod_factor = 2;
  EltwiseReduceMod(result.data(), op.data(), op.size(), modulus,
                   input_mod_factor, output_mod_factor);
  CheckEqual(result, exp_out);
}

}  // namespace hexl
}  // namespace intel
