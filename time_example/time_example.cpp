#include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../hexl/include/hexl/hexl.hpp"
#include "../hexl/include/hexl/util/util.hpp"
#include "../hexl/util/util-internal.hpp"
// #include "../hexl/include/hexl/experimental/fft-like/fft-like.hpp"
// #include "../hexl/include/hexl/experimental/fft-like/fft-like-native.hpp"


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

double BM_EltwiseVectorVectorAddMod(size_t input_size) {  
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);
  auto input2 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);
  intel::hexl::AlignedVector64<uint64_t> output(input_size, 0);

  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseAddMod(output.data(), input1.data(), input2.data(),
                               input_size, modulus);
  });

  return time_taken;
}

double BM_EltwiseVectorScalarAddMod(size_t input_size) {
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);
  uint64_t input2 =
      intel::hexl::GenerateInsecureUniformIntRandomValue(0, modulus);
  intel::hexl::AlignedVector64<uint64_t> output(input_size, 0);

  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseAddMod(output.data(), input1.data(), input2, input_size,
                               modulus);
  });

  return time_taken;
}

double BM_EltwiseCmpAdd(size_t input_size, intel::hexl::CMPINT chosenCMP) {
  uint64_t modulus = 100;

  uint64_t bound =
      intel::hexl::GenerateInsecureUniformIntRandomValue(0, modulus);
  uint64_t diff =
      intel::hexl::GenerateInsecureUniformIntRandomValue(1, modulus - 1);
  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);

  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseCmpAdd(input1.data(), input1.data(), input_size,
                                     chosenCMP, bound, diff);
  });

  return time_taken;
}

double BM_EltwiseCmpSubMod(
    size_t input_size, intel::hexl::CMPINT chosenCMP) { 
  uint64_t modulus = 100;

  uint64_t bound =
      intel::hexl::GenerateInsecureUniformIntRandomValue(1, modulus);
  uint64_t diff =
      intel::hexl::GenerateInsecureUniformIntRandomValue(1, modulus);
  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);

  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseCmpSubMod(input1.data(), input1.data(), input_size,
                                  modulus, chosenCMP, bound, diff);
  });

  return time_taken;
}

double BM_EltwiseFMAModAdd(size_t input_size, bool add) { 
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);
  uint64_t input2 =
      intel::hexl::GenerateInsecureUniformIntRandomValue(0, modulus);
  intel::hexl::AlignedVector64<uint64_t> input3 =
      intel::hexl::GenerateInsecureUniformIntRandomValues(input_size, 0,
                                                          modulus);
  uint64_t* arg3 = add ? input3.data() : nullptr;
  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseFMAMod(input1.data(), input1.data(), input2, arg3,
                               input1.size(), modulus, 1);
  });
  return time_taken;
}

double BM_EltwiseMultMod(size_t input_size, size_t bit_width,
                         size_t input_mod_factor) {
  uint64_t modulus = (1ULL << bit_width) + 7;

  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);
  auto input2 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);
  intel::hexl::AlignedVector64<uint64_t> output(input_size, 2);

  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseMultMod(output.data(), input1.data(), input2.data(),
                                input_size, modulus, input_mod_factor);
  });
  return time_taken;
}

// double BM_EltwiseReduceModInPlace(size_t input_size) {
//   uint64_t modulus = 0xffffffffffc0001ULL;

//   auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(
//       input_size, 0, 100 * modulus);

//   const uint64_t input_mod_factor = modulus;
//   const uint64_t output_mod_factor = 1;

//   double time_taken = TimeFunction([&]() {
//     intel::hexl::EltwiseReduceMod(input1.data(), input1.data(), input_size,
//                                   modulus, input_mod_factor, output_mod_factor);
//   });
//   return time_taken;
// }
double BM_EltwiseReduceModInPlace(size_t input_size) {
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(
      input_size, 0, 100 * modulus);
  intel::hexl::AlignedVector64<uint64_t> output(input_size, 0);

  const uint64_t input_mod_factor = modulus;
  const uint64_t output_mod_factor = 1;

  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseReduceMod(output.data(), input1.data(), input_size,
                                  modulus, input_mod_factor, output_mod_factor);
  });
  return time_taken;
}

double BM_EltwiseVectorVectorSubMod(size_t input_size) {
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);
  auto input2 = intel::hexl::GenerateInsecureUniformIntRandomValues(input_size,
                                                                    0, modulus);
  intel::hexl::AlignedVector64<uint64_t> output(input_size, 0);

  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseSubMod(output.data(), input1.data(), input2.data(),
                               input_size, modulus);
  });
  return time_taken;
}

double BM_NTTInPlace(size_t ntt_size) {
  size_t modulus = intel::hexl::GeneratePrimes(1, 45, true, ntt_size)[0];

  auto input =
      intel::hexl::GenerateInsecureUniformIntRandomValues(ntt_size, 0, modulus);
  intel::hexl::NTT ntt(ntt_size, modulus);

  double time_taken = TimeFunction([&]() {
                        ntt.ComputeForward(input.data(), input.data(), 1, 1);
                      }) +
                      TimeFunction([&]() {
                        ntt.ComputeInverse(input.data(), input.data(), 2, 1);
                      });
  return time_taken;
}

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0]
              << " <num_iterations> <thread_nums(comma-separated)> <input_size> <method>"
              << std::endl;
    return 1;
  }

  int num_iterations = std::stoi(argv[1]);
  std::vector<int> thread_nums = split(argv[2], ',');
  int input_size = std::stoi(argv[3]);
  int method_number = std::stoi(argv[4]);
  std::string method;
  if(method_number>=9 || method_number<0){
    std::cerr
        << "Method Number range from 0 to 8" <<  "0--EltwiseVectorVectorAddMod, "
        << "1 --EltwiseVectorScalarAddMod, " << "2 --EltwiseCmpAdd, "
        << "3 --EltwiseCmpSubMod, " << "4 --EltwiseFMAModAdd, "
        << "5 --EltwiseMultMod, " << "6 --EltwiseReduceModInPlace, "
        << "7 --EltwiseVectorVectorSubMod, "
        << "8 --NTTInPlace, "
        << std::endl;
    return 1;
  }

    // Using a map to store the results
  std::map<std::uint32_t, std::uint64_t> results;
  // std::function<double(size_t)> operation;
      // Initialize the results map

  switch (method_number) {
    case 0:
      method = "BM_EltwiseVectorVectorAddMod";
      break;

    case 1:
      method = "BM_EltwiseVectorScalarAddMod";
      break;

    case 2:
      method = "BM_EltwiseCmpAdd";
      break;

    case 3:
      method = "BM_EltwiseCmpSubMod";
      break;

    case 4:
      method = "BM_EltwiseFMAModAdd";
      break;

    case 5:
      method = "BM_EltwiseMultMod";
      // std::cout << "BM_EltwiseMultMod" << std::endl;
      break;

    case 6:
      method = "BM_EltwiseReduceModInPlace";
      // std::cout << "BM_EltwiseReduceModInPlace" << std::endl;
      break;

    case 7:
      method = "BM_EltwiseVectorVectorSubMod";
      // std::cout << "BM_EltwiseVectorVectorSubMod" << std::endl;
      break;

    case 8:
      method = "BM_NTTInPlace";
      // std::cout << "BM_NTTInPlace" << std::endl;
      break;

    default:
      method = "BM_EltwiseVectorVectorAddMod";
      // std::cout << "Default" << std::endl;
      break;
  }

  // Execute each method for all thread numbers
  for (size_t j = 0; j < thread_nums.size(); ++j) {
    int num_threads = thread_nums[j];
    omp_set_num_threads(num_threads);

    // Initialize the results map
    results[num_threads] = 0.0;

    bool add_choices[] = {false,true};
    int bit_width_choices[] = {48, 60};
    int mod_factor_choices[] = {1, 2, 4};

    for (int i = 0; i < num_iterations; i++) {
      // There are CMPINT possibilities, should be chosen randomly for testing
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis_cmpint(0, 7);  // 8 enum values, from 0 to 7
      std::uniform_int_distribution<> dis_add(0, 1);
      std::uniform_int_distribution<> dis_factor(0, 2);

      intel::hexl::CMPINT chosenCMP =
          static_cast<intel::hexl::CMPINT>(dis_cmpint(gen));

      bool add = add_choices[dis_add(gen)];

      size_t bit_width = bit_width_choices[dis_add(gen)];

      size_t input_mod_factor = mod_factor_choices[dis_factor(gen)];

      if(method_number == 0 || method_number == 1 || method_number == 6 || method_number == 7 || method_number == 8){
        std::function<double(size_t)> operation;
        switch (method_number) {
          case 0:
            operation = BM_EltwiseVectorVectorAddMod;
            // std::cout << "BM_EltwiseVectorVectorAddMod" << std::endl;
            break;

          case 1:
             operation = BM_EltwiseVectorScalarAddMod;
            // std::cout << "BM_EltwiseVectorScalarAddMod" << std::endl;
            break;

          case 6:
            operation = BM_EltwiseReduceModInPlace;
            // std::cout << "BM_EltwiseReduceModInPlace" << std::endl;
            break;

          case 7:
            operation = BM_EltwiseVectorVectorSubMod;
            // std::cout << "BM_EltwiseVectorVectorSubMod" << std::endl;
            break;

          case 8: 
            operation = BM_NTTInPlace;
            input_size = input_size/4096;
                // std::cout << "BM_NTTInPlace" << std::endl;
            break;

          default:
            operation = BM_EltwiseVectorVectorAddMod;
            // std::cout << "Default" << std::endl;
            break;
        }
        results[num_threads] += operation(input_size);
      }

      else if(method_number == 2 || method_number == 3 ){
        std::function<double(size_t, intel::hexl::CMPINT)> operation;
        switch (method_number) {
          case 2:
            operation = BM_EltwiseCmpAdd;
            // std::cout << "BM_EltwiseCmpAdd" << std::endl;
            break;

          case 3:
            operation = BM_EltwiseCmpSubMod;
            // std::cout << "BM_EltwiseCmpSubMod" << std::endl;
            break;

          default:
            operation = BM_EltwiseCmpAdd;
            // std::cout << "Default" << std::endl;
            break;
        }

        results[num_threads] += operation(input_size, chosenCMP);
      }
      else if(method_number == 4){
        std::function<double(size_t, bool)> operation = BM_EltwiseFMAModAdd;
        // std::cout << "BM_EltwiseFMAModAdd" << std::endl;
        results[num_threads] += operation(input_size, add);
      }
      else if (method_number == 5) {
        std::function<double(size_t, size_t, size_t)> operation = BM_EltwiseMultMod;
        // std::cout << "BM_EltwiseMultMod" << std::endl;
        results[num_threads] +=
            operation(input_size, bit_width, input_mod_factor);
      }
        
      }
  }

  return 0;
}
