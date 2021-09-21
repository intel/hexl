// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "hexl/logging/logging.hpp"
#include "hexl/ntt/ntt.hpp"
#include "hexl/number-theory/number-theory.hpp"
#include "ntt/fwd-ntt-avx512.hpp"
#include "ntt/inv-ntt-avx512.hpp"
#include "ntt/ntt-avx512-util.hpp"
#include "ntt/ntt-internal.hpp"
#include "test-util.hpp"
#include "util/cpu-features.hpp"

namespace intel {
namespace hexl {

#ifdef HEXL_HAS_AVX512DQ
TEST(NTT, LoadFwdInterleavedT1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadFwdInterleavedT1(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
  __m512i exp2 = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadInvInterleavedT1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadInvInterleavedT1(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
  __m512i exp2 = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadFwdInterleavedT2) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadFwdInterleavedT2(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  __m512i exp2 = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadInvInterleavedT2) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadInvInterleavedT2(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
  __m512i exp2 = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadFwdInterleavedT4) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadFwdInterleavedT4(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
  __m512i exp2 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, LoadInvInterleavedT4) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  __m512i out1;
  __m512i out2;

  LoadInvInterleavedT4(arg.data(), &out1, &out2);

  __m512i exp1 = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);
  __m512i exp2 = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
  AssertEqual(ExtractValues(out1), ExtractValues(exp1));
  AssertEqual(ExtractValues(out2), ExtractValues(exp2));
}

TEST(NTT, WriteFwdInterleavedT1) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  __m512i arg1 = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i arg2 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

  AlignedVector64<uint64_t> out(16, 0);
  AlignedVector64<uint64_t> exp{8,  0, 9,  1, 10, 2, 11, 3,
                                12, 4, 13, 5, 14, 6, 15, 7};

  WriteFwdInterleavedT1(arg1, arg2, reinterpret_cast<__m512i*>(&out[0]));

  AssertEqual(exp, out);
}

TEST(NTT, WriteInvInterleavedT4) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

  AlignedVector64<uint64_t> arg{0, 1, 2,  3,  4,  5,  6,  7,
                                8, 9, 10, 11, 12, 13, 14, 15};
  __m512i arg1 = _mm512_set_epi64(15, 14, 13, 12, 11, 10, 9, 8);
  __m512i arg2 = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);

  AlignedVector64<uint64_t> out(16, 0);
  AlignedVector64<uint64_t> exp{8,  9,  10, 11, 0, 1, 2, 3,
                                12, 13, 14, 15, 4, 5, 6, 7};

  WriteInvInterleavedT4(arg1, arg2, reinterpret_cast<__m512i*>(&out[0]));

  AssertEqual(exp, out);
}

#ifdef HEXL_HAS_AVX512IFMA
// First parameter is the NTT degree
// Second parameter is the number of bits in the NTT modulus
// Third parameter is whether or not to prefer small primes
class DegreeModulusBoolTest
    : public ::testing::TestWithParam<std::tuple<uint64_t, uint64_t, bool>> {
 protected:
  void SetUp() {}

  void TearDown() {}

 public:
};

TEST_P(DegreeModulusBoolTest, FwdNTTAVX512IFMA) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  bool prefer_small_primes = std::get<2>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, prefer_small_primes, N)[0];

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t trial = 0; trial < num_trials; ++trial) {
    AlignedVector64<uint64_t> input64 =
        GenerateInsecureUniformRandomValues(N, 0, modulus);
    AlignedVector64<uint64_t> input_ifma = input64;
    AlignedVector64<uint64_t> input_ifma_lazy = input64;
    AlignedVector64<uint64_t> exp_output(N, 0);

    // Compute reference
    NTT ntt64(N, modulus);
    ReferenceForwardTransformToBitReverse(input64.data(), N, modulus,
                                          ntt64.GetRootOfUnityPowers().data());

    ForwardTransformToBitReverseAVX512<52>(
        input_ifma.data(), N, ntt64.GetModulus(),
        ntt64.GetAVX512RootOfUnityPowers().data(),
        ntt64.GetAVX512Precon52RootOfUnityPowers().data(), 1, 1);

    // Compute lazy
    ForwardTransformToBitReverseAVX512<52>(
        input_ifma_lazy.data(), N, ntt64.GetModulus(),
        ntt64.GetAVX512RootOfUnityPowers().data(),
        ntt64.GetAVX512Precon52RootOfUnityPowers().data(), 2, 4);
    for (auto& elem : input_ifma_lazy) {
      elem = elem % modulus;
    }

    AssertEqual(input64, input_ifma);
    AssertEqual(input64, input_ifma_lazy);
  }
}

TEST_P(DegreeModulusBoolTest, InvNTTAVX512IFMA) {
  if (!has_avx512ifma) {
    GTEST_SKIP();
  }
  uint64_t N = std::get<0>(GetParam());
  uint64_t modulus_bits = std::get<1>(GetParam());
  bool prefer_small_primes = std::get<2>(GetParam());
  uint64_t modulus = GeneratePrimes(1, modulus_bits, prefer_small_primes, N)[0];

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t trial = 0; trial < num_trials; ++trial) {
    AlignedVector64<uint64_t> input64 =
        GenerateInsecureUniformRandomValues(N, 0, modulus);
    AlignedVector64<uint64_t> input_ifma = input64;
    AlignedVector64<uint64_t> input_ifma_lazy = input64;

    AlignedVector64<uint64_t> exp_output(N, 0);

    // Compute reference
    NTT ntt(N, modulus);
    InverseTransformFromBitReverseRadix2(
        input64.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
        ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

    InverseTransformFromBitReverseAVX512<52>(
        input_ifma.data(), N, ntt.GetModulus(),
        ntt.GetInvRootOfUnityPowers().data(),
        ntt.GetPrecon52InvRootOfUnityPowers().data(), 1, 1);

    // Compute lazy
    InverseTransformFromBitReverseAVX512<52>(
        input_ifma_lazy.data(), N, ntt.GetModulus(),
        ntt.GetInvRootOfUnityPowers().data(),
        ntt.GetPrecon52InvRootOfUnityPowers().data(), 1, 2);
    for (auto& elem : input_ifma_lazy) {
      elem = elem % modulus;
    }

    AssertEqual(input64, input_ifma);
    AssertEqual(input64, input_ifma_lazy);
  }
}

// Test modulus around 50 bits to check IFMA behavior
INSTANTIATE_TEST_SUITE_P(
    NTT, DegreeModulusBoolTest,
    ::testing::Combine(::testing::ValuesIn(AlignedVector64<uint64_t>{
                           1 << 11, 1 << 12, 1 << 13}),
                       ::testing::ValuesIn(AlignedVector64<uint64_t>{48, 49}),
                       ::testing::ValuesIn(std::vector<bool>{false, true})));

#endif

// Checks AVX512 and native forward NTT implementations match
TEST(NTT, FwdNTT_AVX512_32) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t N = 512; N <= 65536; N *= 2) {
    uint64_t modulus = GeneratePrimes(1, 27, true, N)[0];

    NTT ntt(N, modulus);

    for (size_t trial = 0; trial < num_trials; ++trial) {
      AlignedVector64<uint64_t> input =
          GenerateInsecureUniformRandomValues(N, 0, modulus);
      AlignedVector64<uint64_t> input_avx = input;
      AlignedVector64<uint64_t> input_avx_lazy = input;

      ForwardTransformToBitReverseRadix2(
          input.data(), N, modulus, ntt.GetRootOfUnityPowers().data(),
          ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);

      ForwardTransformToBitReverseAVX512<32>(
          input_avx.data(), N, ntt.GetModulus(),
          ntt.GetAVX512RootOfUnityPowers().data(),
          ntt.GetAVX512Precon32RootOfUnityPowers().data(), 2, 1);

      // Compute lazy
      ForwardTransformToBitReverseAVX512<32>(
          input_avx_lazy.data(), N, ntt.GetModulus(),
          ntt.GetAVX512RootOfUnityPowers().data(),
          ntt.GetAVX512Precon32RootOfUnityPowers().data(), 2, 4);
      for (auto& elem : input_avx_lazy) {
        elem = elem % modulus;
      }

      ASSERT_EQ(input, input_avx);
      ASSERT_EQ(input, input_avx_lazy);
    }
  }
}

// Checks AVX512 and native forward NTT implementations match
TEST(NTT, FwdNTT_AVX512_64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t N = 512; N <= 65536; N *= 2) {
    uint64_t modulus = GeneratePrimes(1, 55, true, N)[0];

    NTT ntt(N, modulus);

    for (size_t trial = 0; trial < num_trials; ++trial) {
      AlignedVector64<uint64_t> input =
          GenerateInsecureUniformRandomValues(N, 0, modulus);
      AlignedVector64<uint64_t> input_avx = input;
      AlignedVector64<uint64_t> input_avx_lazy = input;

      ForwardTransformToBitReverseRadix2(
          input.data(), N, modulus, ntt.GetRootOfUnityPowers().data(),
          ntt.GetPrecon64RootOfUnityPowers().data(), 2, 1);

      ForwardTransformToBitReverseAVX512<64>(
          input_avx.data(), N, ntt.GetModulus(),
          ntt.GetAVX512RootOfUnityPowers().data(),
          ntt.GetAVX512Precon64RootOfUnityPowers().data(), 2, 1);

      // Compute lazy
      ForwardTransformToBitReverseAVX512<64>(
          input_avx_lazy.data(), N, ntt.GetModulus(),
          ntt.GetAVX512RootOfUnityPowers().data(),
          ntt.GetAVX512Precon64RootOfUnityPowers().data(), 2, 4);
      for (auto& elem : input_avx_lazy) {
        elem = elem % modulus;
      }

      ASSERT_EQ(input, input_avx);
      ASSERT_EQ(input, input_avx_lazy);
    }
  }
}

// Checks 32-bit AVX512 and native InvNTT implementations match
TEST(NTT, InvNTT_AVX512_32) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t N = 512; N <= 65536; N *= 2) {
    uint64_t modulus = GeneratePrimes(1, 27, true, N)[0];

    NTT ntt(N, modulus);

    for (size_t trial = 0; trial < num_trials; ++trial) {
      AlignedVector64<uint64_t> input =
          GenerateInsecureUniformRandomValues(N, 0, modulus);

      AlignedVector64<uint64_t> input_avx = input;
      AlignedVector64<uint64_t> input_avx_lazy = input;

      InverseTransformFromBitReverseRadix2(
          input.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

      InverseTransformFromBitReverseAVX512<32>(
          input_avx.data(), N, ntt.GetModulus(),
          ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon32InvRootOfUnityPowers().data(), 1, 1);

      // Compute lazy
      InverseTransformFromBitReverseAVX512<32>(
          input_avx_lazy.data(), N, ntt.GetModulus(),
          ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon32InvRootOfUnityPowers().data(), 1, 2);
      for (auto& elem : input_avx_lazy) {
        elem = elem % modulus;
      }

      ASSERT_EQ(input, input_avx);
      ASSERT_EQ(input, input_avx_lazy);
    }
  }
}

// Checks 64-bit AVX512 and native InvNTT implementations match
TEST(NTT, InvNTT_AVX512_64) {
  if (!has_avx512dq) {
    GTEST_SKIP();
  }

#ifdef HEXL_DEBUG
  size_t num_trials = 1;
#else
  size_t num_trials = 20;
#endif

  for (size_t N = 512; N <= 65536; N *= 2) {
    uint64_t modulus = GeneratePrimes(1, 55, true, N)[0];

    NTT ntt(N, modulus);

    for (size_t trial = 0; trial < num_trials; ++trial) {
      AlignedVector64<uint64_t> input =
          GenerateInsecureUniformRandomValues(N, 0, modulus);
      AlignedVector64<uint64_t> input_avx = input;
      AlignedVector64<uint64_t> input_avx_lazy = input;

      InverseTransformFromBitReverseRadix2(
          input.data(), N, modulus, ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

      InverseTransformFromBitReverseAVX512<64>(
          input_avx.data(), N, ntt.GetModulus(),
          ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 1);

      // Compute lazy
      InverseTransformFromBitReverseAVX512<64>(
          input_avx_lazy.data(), N, ntt.GetModulus(),
          ntt.GetInvRootOfUnityPowers().data(),
          ntt.GetPrecon64InvRootOfUnityPowers().data(), 1, 2);
      for (auto& elem : input_avx_lazy) {
        elem = elem % modulus;
      }

      ASSERT_EQ(input, input_avx);
      ASSERT_EQ(input, input_avx_lazy);
    }
  }
}
#endif

}  // namespace hexl
}  // namespace intel
