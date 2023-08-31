// #include <omp.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../hexl/include/hexl/hexl.hpp"
// #include "../hexl/include/hexl/util/util.hpp"
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

double BM_EltwiseCmpSubMod(size_t input_size, intel::hexl::CMPINT chosenCMP) {
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

double BM_EltwiseReduceModInPlace(size_t input_size) {
  uint64_t modulus = 0xffffffffffc0001ULL;

  auto input1 = intel::hexl::GenerateInsecureUniformIntRandomValues(
      input_size, 0, 100 * modulus);

  const uint64_t input_mod_factor = modulus;
  const uint64_t output_mod_factor = 1;

  double time_taken = TimeFunction([&]() {
    intel::hexl::EltwiseReduceMod(input1.data(), input1.data(), input_size,
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

// double BM_FwdFFTLikeRd2InPlaceSmall(const size_t fft_like_size) {
//   const size_t bound = 1 << 30;
//   const double scale = 10;
//   const double scalar = scale / static_cast<double>(fft_like_size);
//   intel::hexl::FFTLike fft_like(fft_like_size, nullptr);
//
//   intel::hexl::AlignedVector64<std::complex<double>> input(fft_like_size);
//   for (size_t i = 0; i < fft_like_size; i++) {
//     input[i] = std::complex<double>(
//         intel::hexl::GenerateInsecureUniformRealRandomValue(0, bound),
//         intel::hexl::GenerateInsecureUniformRealRandomValue(0, bound));
//   }
//
//   intel::hexl::AlignedVector64<std::complex<double>> root_powers =
//       fft_like.GetComplexRootsOfUnity();
//
//   double time_taken = TimeFunction([&]() {
//     intel::hexl::Forward_FFTLike_ToBitReverseRadix2(
//         input.data(), input.data(), root_powers.data(), fft_like_size,
//         &scalar);
//   });
//   return time_taken;
// }
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
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0]
              << " <num_iterations> <input_size>"
              << std::endl;
    return 1;
  }

  int num_iterations = std::stoi(argv[1]);
  std::vector<int> input_size = split(argv[2], ',');

  // Using a map to store the results
  std::map<std::string, std::vector<double>> results;

  // Initialize the results map
  results["BM_EltwiseVectorVectorAddMod"] =
      std::vector<double>(input_size.size(), 0.0);
  results["BM_EltwiseVectorScalarAddMod"] =
      std::vector<double>(input_size.size(), 0.0);
  results["BM_EltwiseCmpAdd"] = std::vector<double>(input_size.size(), 0.0);
  results["BM_EltwiseCmpSubMod"] = std::vector<double>(input_size.size(), 0.0);
  results["BM_EltwiseFMAModAdd"] = std::vector<double>(input_size.size(), 0.0);
  results["BM_EltwiseMultMod"] = std::vector<double>(input_size.size(), 0.0);
  results["BM_EltwiseReduceModInPlace"] =
      std::vector<double>(input_size.size(), 0.0);
  results["BM_EltwiseVectorVectorSubMod"] =
      std::vector<double>(input_size.size(), 0.0);
  results["BM_NTTInPlace"] = std::vector<double>(input_size.size(), 0.0);

  // Execute each method for all thread numbers
  for (size_t j = 0; j < input_size.size(); ++j) {
    int size = input_size[j];
    // omp_set_num_threads(num_threads);

    results["BM_EltwiseVectorVectorAddMod"][j] = 0;
    results["BM_EltwiseVectorScalarAddMod"][j] = 0;
    results["BM_EltwiseCmpAdd"][j] = 0;
    results["BM_EltwiseCmpSubMod"][j] = 0;
    results["BM_EltwiseFMAModAdd"][j] = 0;
    results["BM_EltwiseMultMod"][j] = 0;
    results["BM_EltwiseReduceModInPlace"][j] = 0;
    results["BM_EltwiseVectorVectorSubMod"][j] = 0;
    results["BM_NTTInPlace"][j] = 0;

    bool add_choices[] = {false, true};
    int bit_width_choices[] = {48, 60};
    int mod_factor_choices[] = {1, 2, 4};

    for (int i = 0; i < num_iterations; i++) {
      // There are CMPINT possibilities, should be chosen randomly for testing
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dis_cmpint(
          0, 7);  // 8 enum values, from 0 to 7
      std::uniform_int_distribution<> dis_add(0, 1);
      std::uniform_int_distribution<> dis_factor(0, 2);

      intel::hexl::CMPINT chosenCMP =
          static_cast<intel::hexl::CMPINT>(dis_cmpint(gen));

      bool add = add_choices[dis_add(gen)];

      size_t bit_width = bit_width_choices[dis_add(gen)];

      size_t input_mod_factor = mod_factor_choices[dis_factor(gen)];

      results["BM_EltwiseVectorVectorAddMod"][j] +=
          BM_EltwiseVectorVectorAddMod(size);
      results["BM_EltwiseVectorScalarAddMod"][j] +=
          BM_EltwiseVectorScalarAddMod(size);
      results["BM_EltwiseCmpAdd"][j] += BM_EltwiseCmpAdd(size, chosenCMP);
      results["BM_EltwiseCmpSubMod"][j] +=
          BM_EltwiseCmpSubMod(size, chosenCMP);
      results["BM_EltwiseFMAModAdd"][j] += BM_EltwiseFMAModAdd(size, add);
      results["BM_EltwiseMultMod"][j] +=
          BM_EltwiseMultMod(size, bit_width, input_mod_factor);
      results["BM_EltwiseReduceModInPlace"][j] +=
          BM_EltwiseReduceModInPlace(size);
      results["BM_EltwiseVectorVectorSubMod"][j] +=
          BM_EltwiseVectorVectorSubMod(size);
      results["BM_NTTInPlace"][j] += BM_NTTInPlace(size/4096);
    }
  }

  // Print the table

  // Print headers
  std::cout << std::left << std::setw(40) << "Method";
  for (int size : input_size) {
    std::cout << std::setw(20) << ("Input_size=" + std::to_string(size));
  }
  std::cout << std::endl;

  // Print results
  for (auto& [method, times] : results) {
    std::cout << std::left << std::setw(40) << method;
    for (double time : times) {
      std::cout << std::setw(20) << time/num_iterations;
    }
    std::cout << std::endl;
  }

  return 0;
}
