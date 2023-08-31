// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <iostream>
#include <vector>

#include "hexl/hexl.hpp"
#include "../hexl/util/util-internal.hpp"

#include <chrono>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <omp.h>


template <typename Func>
double TimeFunction(Func&& f) {
    auto start_time = std::chrono::high_resolution_clock::now();
    f();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    return duration.count();
}

std::vector<int> split(const std::string& s, char delimiter) {
    std::vector<int> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
      tokens.push_back(std::stoi(token));
    }
    return tokens;
}

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

double ExampleEltwiseVectorVectorAddMod() {
//  std::cout << "Running ExampleEltwiseVectorVectorAddMod...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> op2{1, 3, 5, 7, 2, 4, 6, 8};
  uint64_t modulus = 10;
  std::vector<uint64_t> exp_out{2, 5, 8, 1, 7, 0, 3, 6};
    
    double time_taken = TimeFunction([&]() {
        intel::hexl::EltwiseAddMod(op1.data(), op1.data(), op2.data(), op1.size(), modulus);
    });

  CheckEqual(op1, exp_out);
//  std::cout << "Done running ExampleEltwiseVectorVectorAddMod\n";
    return time_taken;
}

double ExampleEltwiseVectorScalarAddMod() {
//  std::cout << "Running ExampleEltwiseVectorScalarAddMod...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t op2{3};
  uint64_t modulus = 10;
  std::vector<uint64_t> exp_out{4, 5, 6, 7, 8, 9, 0, 1};

    double time_taken = TimeFunction([&]() {
        intel::hexl::EltwiseAddMod(op1.data(), op1.data(), op2, op1.size(), modulus);
    });

  CheckEqual(op1, exp_out);
//  std::cout << "Done running ExampleEltwiseVectorScalarAddMod\n";
    return time_taken;
}

double ExampleEltwiseCmpAdd() {
//  std::cout << "Running ExampleEltwiseCmpAdd...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t cmp = 3;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 9, 10, 11, 12, 13};
    
    double time_taken = TimeFunction([&]() {
        intel::hexl::EltwiseCmpAdd(op1.data(), op1.data(), op1.size(), intel::hexl::CMPINT::NLE, cmp, diff);
    });


  CheckEqual(op1, exp_out);
//  std::cout << "Done running ExampleEltwiseCmpAdd\n";
    return time_taken;
}

double ExampleEltwiseCmpSubMod() {
//  std::cout << "Running ExampleEltwiseCmpSubMod...\n";

  std::vector<uint64_t> op1{1, 2, 3, 4, 5, 6, 7};
  uint64_t bound = 4;
  uint64_t diff = 5;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 0, 1, 2};

  uint64_t modulus = 10;

    double time_taken = TimeFunction([&]() {
        intel::hexl::EltwiseCmpSubMod(op1.data(), op1.data(), op1.size(), modulus, intel::hexl::CMPINT::NLE, bound, diff);
    });
    
  CheckEqual(op1, exp_out);
//  std::cout << "Done running ExampleEltwiseCmpSubMod\n";
    return time_taken;
}

double ExampleEltwiseFMAMod() {
//  std::cout << "Running ExampleEltwiseFMAMod...\n";

  std::vector<uint64_t> arg1{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t arg2 = 1;
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 5, 6, 7, 8, 9};
  uint64_t modulus = 769;
    
    double time_taken;
    time_taken = TimeFunction([&]() {
        intel::hexl::EltwiseFMAMod(arg1.data(), arg1.data(), arg2, nullptr, arg1.size(), modulus, 1);
    });
  
  CheckEqual(arg1, exp_out);
//  std::cout << "Done running ExampleEltwiseFMAMod\n";
    return time_taken;
}

double ExampleEltwiseMultMod() {
//  std::cout << "Running ExampleEltwiseMultMod...\n";

  std::vector<uint64_t> op1{2, 4, 3, 2};
  std::vector<uint64_t> op2{2, 1, 2, 0};
  std::vector<uint64_t> exp_out{4, 4, 6, 0};

  uint64_t modulus = 769;

    double time_taken = TimeFunction([&]() {
        intel::hexl::EltwiseMultMod(op1.data(), op1.data(), op2.data(), op1.size(), modulus, 1);
    });
  
  CheckEqual(op1, exp_out);
//  std::cout << "Done running ExampleEltwiseMultMod\n";
    return time_taken;
}

double ExampleNTT() {
//  std::cout << "Running ExampleNTT...\n";

  uint64_t N = 8;
  uint64_t modulus = 769;
  std::vector<uint64_t> arg{1, 2, 3, 4, 5, 6, 7, 8};
  auto exp_out = arg;
  intel::hexl::NTT ntt(N, modulus);
    
    double time_taken = TimeFunction([&]() {
        ntt.ComputeForward(arg.data(), arg.data(), 1, 1);
    });
    time_taken += TimeFunction([&]() {
        ntt.ComputeInverse(arg.data(), arg.data(), 1, 1);
    });

  CheckEqual(arg, exp_out);
//  std::cout << "Done running ExampleNTT\n";
    return time_taken;
}

double ExampleEltwiseReduceMod() {
//  std::cout << "Running ExampleReduceMod...\n";

  uint64_t modulus = 5;
  std::vector<uint64_t> arg{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<uint64_t> exp_out{1, 2, 3, 4, 0, 1, 2, 3};
  std::vector<uint64_t> result{0, 0, 0, 0, 0, 0, 0, 0};
  
    double time_taken = TimeFunction([&]() {
        intel::hexl::EltwiseReduceMod(result.data(), arg.data(), arg.size(), modulus,2, 1);
    });
    
  CheckEqual(result, exp_out);
//  std::cout << "Done running ExampleReduceMod\n";
    return time_taken;
}

// double BM_EltwiseVectorVectorAddMod(size_t input_size) {  //  NOLINT
//   uint64_t modulus = 0xffffffffffc0001ULL;

//   auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
//                                                                     0, modulus);
//   auto input2 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
//                                                                     0, modulus);
//   intel::hexl::AlignedVector64<uint64_t> output(input_size, 0);

//   double time_taken = TimeFunction([&]() {
//         intel::hexl::EltwiseAddMod(output.data(), input1.data(), input2.data(), input_size, modulus);
//   });

//   return time_taken;
// }

// double BM_EltwiseVectorScalarAddMod(size_t input_size) {
//   uint64_t modulus = 0xffffffffffc0001ULL;

//   auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size, 0, modulus);
//   uint64_t input2 = intel::hexl::GenerateInsecureUniformIntRandomValue(0, modulus);
//   intel::hexl::AlignedVector64<uint64_t> output(input_size, 0);

//   double time_taken = TimeFunction([&]() {
//         intel::hexl::EltwiseAddMod(output.data(), input1.data(), input2, input_size, modulus);
//   });

//   return time_taken;
// }


int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <num_iterations> <num_threads> <input_size>" << std::endl;
        return 1;
    }

    int num_iterations = std::stoi(argv[1]);
    std::vector<int> thread_nums = split(argv[2], ',');
    int input_size = std::stoi(argv[3]);

    for (size_t j = 0; j < thread_nums.size(); ++j) {
        int num_threads = thread_nums[j];
        omp_set_num_threads(num_threads);

        double t_EltwiseVectorVectorAddMod = 0.0;
        double t_EltwiseVectorScalarAddMod = 0.0;
        double t_EltwiseCmpAdd = 0.0;
        double t_EltwiseCmpSubMod = 0.0;
        double t_EltwiseFMAMod = 0.0;
        double t_EltwiseMultMod = 0.0;
        double t_NTT = 0.0;
        double t_EltwiseReduceMod = 0.0;
        // double t_BM_EltwiseVectorVectorAddMod = 0.0;
        // double t_BM_EltwiseVectorScalarAddMod = 0.0;

        // Rest of your code
        for (int i = 0; i < num_iterations; i++) {

          // computation code for each iteration
          t_EltwiseVectorVectorAddMod += ExampleEltwiseVectorVectorAddMod();
          t_EltwiseVectorScalarAddMod += ExampleEltwiseVectorScalarAddMod();
          t_EltwiseCmpAdd += ExampleEltwiseCmpAdd();
          t_EltwiseCmpSubMod += ExampleEltwiseCmpSubMod();
          t_EltwiseFMAMod += ExampleEltwiseFMAMod();
          t_EltwiseMultMod += ExampleEltwiseMultMod();
          t_NTT += ExampleNTT();
          t_EltwiseReduceMod += ExampleEltwiseReduceMod();
          // t_BM_EltwiseVectorVectorAddMod +=
          // BM_EltwiseVectorVectorAddMod(input_size);
          // t_BM_EltwiseVectorScalarAddMod +=
          // BM_EltwiseVectorScalarAddMod(input_size);
        }

        std::cout << "Thread Number: " << num_threads
                  << "  after itertating: " << num_iterations << "  times  "<< std::endl;

        // Calculate the width for columns based on the longest label
        int labelWidth = 40;
        int timeWidth = 20;

        std::cout << std::left << std::setw(labelWidth) << "Operation"
                  << std::setw(timeWidth) << "Time (ms)" << std::endl;
        std::cout << std::setfill('-') << std::setw(labelWidth + timeWidth)
                  << "-" << std::setfill(' ') << std::endl;
        std::cout << std::left << std::setw(labelWidth)
                  << "EltwiseVectorVectorAddMod" << std::setw(timeWidth)
                  << t_EltwiseVectorVectorAddMod << std::endl;
        std::cout << std::left << std::setw(labelWidth)
                  << "EltwiseVectorScalarAddMod" << std::setw(timeWidth)
                  << t_EltwiseVectorScalarAddMod << std::endl;
        std::cout << std::left << std::setw(labelWidth) << "EltwiseCmpAdd"
                  << std::setw(timeWidth) << t_EltwiseCmpAdd << std::endl;
        std::cout << std::left << std::setw(labelWidth) << "EltwiseCmpSubMod"
                  << std::setw(timeWidth) << t_EltwiseCmpSubMod << std::endl;
        std::cout << std::left << std::setw(labelWidth) << "EltwiseFMAMod"
                  << std::setw(timeWidth) << t_EltwiseFMAMod << std::endl;
        std::cout << std::left << std::setw(labelWidth) << "EltwiseMultMod"
                  << std::setw(timeWidth) << t_EltwiseMultMod << std::endl;
        std::cout << std::left << std::setw(labelWidth) << "EltwiseReduceMod"
                  << std::setw(timeWidth) << t_EltwiseReduceMod << std::endl;
        std::cout << std::left << std::setw(labelWidth) << "NTT"
                  << std::setw(timeWidth) << t_NTT << std::endl;
        // std::cout << std::left << std::setw(labelWidth) <<
        // "BM_EltwiseVectorVectorAddMod" << std::setw(timeWidth) <<
        // t_BM_EltwiseVectorVectorAddMod << std::endl; std::cout << std::left
        // << std::setw(labelWidth) << "BM_EltwiseVectorScalarAddMod" <<
        // std::setw(timeWidth) << t_BM_EltwiseVectorScalarAddMod << std::endl;
        std::cout << "******************************************************" << std::endl;
    }

      return 0;
    }
