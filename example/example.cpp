// Copyright (C) 2020-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <vector>

#include "hexl/hexl.hpp"

bool CheckEqual(const std::vector<uint64_t>& x,
                const std::vector<uint64_t>& y) {
  if (x.size() != y.size()) {
    std::cout << "Not equal in size\n";
    return false;
  }
  uint64_t N = x.size();
  bool is_match = true;
  for (size_t i = 0; i < N; ++i) {
    if (x[i] != y[i]) {
      std::cout << "Not equal at index " << i << "\n";
      is_match = false;
    }
  }
  return is_match;
}

void ExampleEltwiseVectorVectorAddMod() {
  std::cout << "Running ExampleEltwiseVectorVectorAddMod...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 2, 4, 6, 8};
  uint64_t modulus = 10;
  std::vector<uint64_t> exp_out{2, 5, 8, 1, 7, 0, 3, 6};

  intel::hexl::EltwiseAddMod(op1.data(), op1.data(), op2.data(), op1.size(),
                             modulus);

  CheckEqual(op1, exp_out);
  std::cout << "Done running ExampleEltwiseVectorVectorAddMod\n";
}

void ExampleEltwiseVectorScalarAddMod() {
  std::cout << "Running ExampleEltwiseVectorScalarAddMod...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t op2{3};
  uint64_t modulus = 10;
  std::vector<uint64_t> exp_out{4, 5, 6, 7, 8, 9, 0, 1};

  intel::hexl::EltwiseAddMod(op1.data(), op1.data(), op2, op1.size(), modulus);

  CheckEqual(op1, exp_out);
  std::cout << "Done running ExampleEltwiseVectorScalarAddMod\n";
}

void ExampleEltwiseCmpAdd() {
  std::cout << "Running ExampleEltwiseCmpAdd...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t cmp = 3;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 9, 10, 11, 12, 13};

  intel::hexl::EltwiseCmpAdd(op1.data(), op1.data(), op1.size(),
                             intel::hexl::CMPINT::NLE, cmp, diff);

  CheckEqual(op1, exp_out);
  std::cout << "Done running ExampleEltwiseCmpAdd\n";
}

void ExampleEltwiseCmpSubMod() {
  std::cout << "Running ExampleEltwiseCmpSubMod...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7};
  uint64_t bound = 4;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 0, 1, 2};

  uint64_t modulus = 10;

  intel::hexl::EltwiseCmpSubMod(op1.data(), op1.data(), op1.size(), modulus,
                                intel::hexl::CMPINT::NLE, bound, diff);
  CheckEqual(op1, exp_out);
  std::cout << "Done running ExampleEltwiseCmpSubMod\n";
}

void ExampleEltwiseFMAMod() {
  std::cout << "Running ExampleEltwiseFMAMod...\n";

  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t arg2 = 1;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t modulus = 769;

  intel::hexl::EltwiseFMAMod(arg1.data(), arg1.data(), arg2, nullptr,
                             arg1.size(), modulus, 1);
  CheckEqual(arg1, exp_out);
  std::cout << "Done running ExampleEltwiseFMAMod\n";
}

void ExampleEltwiseMultMod() {
  std::cout << "Running ExampleEltwiseMultMod...\n";

  std::vector<uint64_t> op1{2, 4, 3, 2};
  std::vector<uint64_t> op2{2, 1, 2, 0};
  std::vector<uint64_t> exp_out{4, 4, 6, 0};

  uint64_t modulus = 769;

  intel::hexl::EltwiseMultMod(op1.data(), op1.data(), op2.data(), op1.size(),
                              modulus, 1);
  CheckEqual(op1, exp_out);
  std::cout << "Done running ExampleEltwiseMultMod\n";
}

void ExampleNTT() {
  std::cout << "Running ExampleNTT...\n";

  uint64_t N = 8;
  uint64_t modulus = 769;
  std::vector<uint64_t> arg{1, 2, 3, 4, 5, 6, 7, 8};
  auto exp_out = arg;
  intel::hexl::NTT ntt(N, modulus);

  ntt.ComputeForward(arg.data(), arg.data(), 1, 1);
  ntt.ComputeInverse(arg.data(), arg.data(), 1, 1);

  CheckEqual(arg, exp_out);
  std::cout << "Done running ExampleNTT\n";
}

void ExampleReduceMod() {
  std::cout << "Running ExampleReduceMod...\n";

  uint64_t modulus = 5;
  std::vector<uint64_t> arg{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 0, 1, 2, 3};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  intel::hexl::EltwiseReduceMod(result.data(), arg.data(), arg.size(), modulus,
                                2, 1);

  CheckEqual(result, exp_out);
  std::cout << "Done running ExampleReduceMod\n";
}

int main() {
  ExampleEltwiseVectorVectorAddMod();
  ExampleEltwiseVectorScalarAddMod();
  ExampleEltwiseCmpAdd();
  ExampleEltwiseCmpSubMod();
  ExampleEltwiseFMAMod();
  ExampleEltwiseMultMod();
  ExampleNTT();
  ExampleReduceMod();

  return 0;
}
