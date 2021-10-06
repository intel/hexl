// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/util/aligned-allocator.hpp"
#include "hexl/util/check.hpp"
#include "hexl/util/compiler.hpp"

namespace intel {
namespace hexl {

// First parameter is the NTT degree
// Second parameter is the number of bits in the NTT modulus
// Third parameter is whether or not to prefer small primes
class DegreeModulusBoolTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, bool>> {
 protected:
  void SetUp() override {
    m_N = std::get<0>(GetParam());
    m_modulus_bits = std::get<1>(GetParam());
    m_prefer_small_primes = std::get<2>(GetParam());
    m_modulus =
        GeneratePrimes(1, m_modulus_bits, m_prefer_small_primes, m_N)[0];
    m_ntt = NTT(m_N, m_modulus);

#ifdef HEXL_DEBUG
    m_num_trials = 1;
#else
    m_num_trials = 10;
#endif
  }

  void TearDown() override {}

 public:
  uint64_t m_N;
  uint64_t m_modulus_bits;
  bool m_prefer_small_primes;
  uint64_t m_modulus;
  NTT m_ntt;

  uint64_t m_num_trials;
};

// Parameters = (degree, modulus, input, expected_output)
class DegreeModulusInputOutput
    : public ::testing::TestWithParam<std::tuple<
          uint64_t, uint64_t, std::vector<uint64_t>, std::vector<uint64_t>>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

}  // namespace hexl
}  // namespace intel
