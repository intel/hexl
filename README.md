[![Build and Test](https://github.com/intel/hexl/actions/workflows/github-ci.yml/badge.svg?branch=main)](https://github.com/intel/hexl/actions/workflows/github-ci.yml)

# Intel Homomorphic Encryption Acceleration Library (HEXL)
Intel:registered: HEXL is an open-source library which provides efficient implementations of integer arithmetic on Galois fields. Such arithmetic is prevalent in cryptography, particularly in homomorphic encryption (HE) schemes. Intel HEXL targets integer arithmetic with word-sized primes, typically 30-60 bits. Intel HEXL provides an API for 64-bit unsigned integers and targets Intel CPUs. For more details on Intel HEXL, see our [whitepaper](https://arxiv.org/abs/2103.16400.pdf). For tips on best performance, see [Performance](#performance)

## Contents
- [Intel Homomorphic Encryption Acceleration Library (HEXL)](#intel-homomorphic-encryption-acceleration-library-hexl)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Building Intel HEXL](#building-intel-hexl)
    - [Dependencies](#dependencies)
    - [Compile-time options](#compile-time-options)
    - [Compiling Intel HEXL](#compiling-intel-hexl)
      - [Linux and Mac](#linux-and-mac)
      - [Windows](#windows)
  - [Performance](#performance)
  - [Testing Intel HEXL](#testing-intel-hexl)
  - [Benchmarking Intel HEXL](#benchmarking-intel-hexl)
  - [Using Intel HEXL](#using-intel-hexl)
  - [Debugging](#debugging)
  - [Threading](#threading)
- [Community Adoption](#community-adoption)
- [Documentation](#documentation)
- [Contributing](#contributing)
  - [Repository layout](#repository-layout)
- [Citing Intel HEXL](#citing-intel-hexl)
    - [Version 1.2](#version-12)
    - [Version 1.1](#version-11)
    - [Version 1.0](#version-10)
- [Contributors](#contributors)

## Introduction
Many cryptographic applications, particularly homomorphic encryption (HE), rely on integer polynomial arithmetic in a finite field. HE, which enables computation on encrypted data, typically uses polynomials with degree `N` a power of two roughly in the range `N=[2^{10}, 2^{17}]`. The coefficients of these polynomials are in a finite field with a word-sized prime, `q`, up to `q`~62 bits. More precisely, the polynomials live in the ring `Z_q[X]/(X^N + 1)`. That is, when adding or multiplying two polynomials, each coefficient of the result is reduced by the prime modulus `q`. When multiplying two polynomials, the resulting polynomials of degree `2N` is additionally reduced by taking the remainder when dividing by `X^N+1`.

 The primary bottleneck in many HE applications is polynomial-polynomial multiplication in `Z_q[X]/(X^N + 1)`. For efficient implementation, Intel HEXL implements the negacyclic number-theoretic transform (NTT). To multiply two polynomials, `q_1(x), q_2(x)` using the NTT, we perform the FwdNTT on the two input polynomials, then perform an element-wise modular multiplication, and perform the InvNTT on the result.

Intel HEXL implements the following functions:
- The forward and inverse negacyclic number-theoretic transform (NTT)
- Element-wise vector-vector modular multiplication
- Element-wise vector-scalar modular multiplication with optional addition
- Element-wise modular multiplication

For each function, the library implements one or several Intel(R) AVX-512 implementations, as well as a less performant, more readable native C++ implementation. Intel HEXL will automatically choose the best implementation for the given CPU Intel(R) AVX-512 feature set. In particular, when the modulus `q` is less than `2^{50}`, the AVX512IFMA instruction set available on Intel IceLake server and IceLake client will provide a more efficient implementation.

For additional functionality, see the public headers, located in `include/hexl`

## Building Intel HEXL

Intel HEXL can be built in several ways. Intel HEXL has been uploaded to the [Microsoft vcpkg](https://github.com/microsoft/vcpkg) C++ package manager, which supports Linux, macOS, and Windows builds. See the vcpkg repository for instructions to build Intel HEXL with vcpkg, e.g. run `vcpkg install hexl`. There may be some delay in uploading latest release ports to vcpkg. Intel HEXL provides port files to build the latest version with vcpkg. For a static build, run `vcpkg install hexl --overlay-ports=/path/to/hexl/port/hexl --head`. For dynamic build, use the custom triplet file and run `vcpkg install hexl:hexl-dynamic-build --overlay-ports=/path/to/hexl/port/hexl --head --overlay-triplets=/path/to/hexl/port/hexl`. For detailed explanation, see [instruction](https://devblogs.microsoft.com/cppblog/registries-bring-your-own-libraries-to-vcpkg/) for building vcpkg port using overlays and use of [custom triplet](https://github.com/microsoft/vcpkg/blob/master/docs/examples/overlay-triplets-linux-dynamic.md#building-dynamic-libraries-on-linux) provided by vcpkg.

Intel HEXL also supports a build using the CMake build system. See below for the instructions to build Intel HEXL from source using CMake.

### Dependencies
We have tested Intel HEXL on the following operating systems:
- Ubuntu 20.04
- macOS 10.15
- Microsoft Windows 10

Intel HEXL requires the following dependencies:

| Dependency  | Version                                      |
|-------------|----------------------------------------------|
| CMake       | >= 3.13                                      |
| Compiler    | gcc >= 7.0, clang++ >= 5.0, MSVC >= 2019     |


### Compile-time options
In addition to the standard CMake build options, Intel HEXL supports several compile-time flags to configure the build.
For convenience, they are listed below:

| CMake option                  | Values   | Default |                                                             |
| ------------------------------| ---------| --------| ----------------------------------------------------------- |
| HEXL_BENCHMARK                | ON / OFF | ON      | Set to ON to enable benchmark suite via Google benchmark    |
| HEXL_COVERAGE                 | ON / OFF | OFF     | Set to ON to enable coverage report of unit-tests           |
| HEXL_SHARED_LIB               | ON / OFF | OFF     | Set to ON to enable building shared library                 |
| HEXL_DOCS                     | ON / OFF | OFF     | Set to ON to enable building of documentation               |
| HEXL_TESTING                  | ON / OFF | ON      | Set to ON to enable building of unit-tests                  |
| HEXL_TREAT_WARNING_AS_ERROR   | ON / OFF | OFF     | Set to ON to treat all warnings as error                    |

### Compiling Intel HEXL
To compile Intel HEXL from source code, first clone the repository and change directories to the where the source has been cloned.
#### Linux and Mac
The instructions to build Intel HEXL are common to Linux and MacOS.

Then, to configure the build, call
```bash
cmake -S . -B build
```
adding the desired compile-time options with a `-D` flag. For instance, to build Intel HEXL as a shared library, call
```bash
cmake -S . -B build -DHEXL_SHARED_LIB=ON
```

Then, to build Intel HEXL, call
```bash
cmake --build build
```
This will build the Intel HEXL library in the `build/hexl/lib/` directory.

To install Intel HEXL to the installation directory, run
```bash
cmake --install build
```
To use a non-standard installation directory, configure the build with
```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/path/to/install
```
before proceeding with the build and installation directions above.

#### Windows
To compile Intel HEXL on Windows using Visual Studio in Release mode, configure the build via
```bash
cmake -S . -B build -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release
```
adding the desired compile-time options with a `-D` flag (see [Compile-time options](#compile-time-options)).

To specify the desired build configuration, pass either `--config Debug` or `--config Release` to the build step and install steps. For instance, to build Intel HEXL in Release mode, call
```bash
cmake --build build --config Release
```
This will build the Intel HEXL library in the `build/hexl/lib/` or `build/hexl/Release/lib` directory.

To install Intel HEXL to the installation directory, run
```bash
cmake --build build --target install --config Release
```
To use a non-standard installation directory, configure the build with
```bash
cmake -S . -B build -G "Visual Studio 16 2019" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/path/to/install
```
before proceeding with the build and installation directions above.

## Performance
For best performance, we recommend using Intel HEXL on a Linux system with the clang++-12 compiler. We also recommend using a processor with Intel AVX512DQ support, with best performance on processors supporting Intel AVX512-IFMA52. To determine if your processor supports AVX512-IFMA52, simply look for `-- Setting HEXL_HAS_AVX512IFMA` printed during the configure step.

See the below table for setting the modulus `q` for best performance.
| Instruction Set  | Bound on modulus `q` |
|------------------|----------------------|
| AVX512-DQ        | `q < 2^30`           |
| AVX512-IFMA52    | `q < 2^50`           |

Some speedup is still expected for moduli `q > 2^30` using the AVX512-DQ instruction set.

## Testing Intel HEXL
To run a set of unit tests via [Googletest](https://github.com/google/googletest), configure and build Intel HEXL with `-DHEXL_TESTING=ON` (see [Compile-time options](#compile-time-options)).
Then, run
```bash
cmake --build build --target unittest
```
The unit-test executable itself is located at `build/test/unit-test` on Linux and Mac, and at `build\test\Release\unit-test.exe` or `build\test\Debug\unit-test.exe` on Windows.
## Benchmarking Intel HEXL
To run a set of benchmarks via [Google benchmark](https://github.com/google/benchmark), configure and build Intel HEXL with `-DHEXL_BENCHMARK=ON` (see [Compile-time options](#compile-time-options)).
Then, run
```bash
cmake --build build --target bench
```
On Windows, run
```bash
cmake --build build --target bench --config Release
```

The benchmark executable itself is located at `build/benchmark/bench_hexl` on Linux and Mac, and at `build\benchmark\Debug\bench_hexl.exe` or `build\benchmark\Release\bench_hexl.exe` on Windows.

## Using Intel HEXL
The `example` folder has an example of using Intel HEXL in a third-party project.

## Debugging
For optimal performance, Intel HEXL does not perform input validation. In many cases the time required for the validation would be longer than the execution of the function itself. To debug Intel HEXL, configure and build Intel HEXL with `-DCMAKE_BUILD_TYPE=Debug` (see [Compile-time options](#compile-time-options)). This will generate a debug version of the library, e.g. `libhexl_debug.a`, that can be used to debug the execution. In Debug mode, Intel HEXL will also link against [Address Sanitizer](https://github.com/google/sanitizers/wiki/AddressSanitizer).

**Note**, enabling `CMAKE_BUILD_TYPE=Debug` will result in a significant runtime overhead.

To enable verbose logging for the benchmarks or unit-tests in a Debug build, add the log level as a command-line argument, e.g. `build/benchmark/bench_hexl --v=9`. See [easyloggingpp's documentation](https://github.com/amrayn/easyloggingpp#application-arguments) for more details.

## Threading
Intel HEXL is single-threaded and thread-safe.

# Community Adoption

Intel HEXL has been integrated to the following homomorphic encryption libraries:
- [HElib](https://github.com/homenc/HElib)
- [Microsoft SEAL](https://github.com/microsoft/SEAL)
- [PALISADE](https://gitlab.com/palisade/palisade-release)

See also the [Intel Homomorphic Encryption Toolkit](https://github.com/intel/he-toolkit) for example uses cases using HEXL.

Please let us know if you are aware of any other uses of Intel HEXL.

# Documentation
Intel HEXL supports documentation via Doxygen. See [https://intel.github.io/hexl](https://intel.github.io/hexl) for the latest Doxygen documentation.

To build documentation, first install `doxygen` and `graphviz`, e.g.
```bash
sudo apt-get install doxygen graphviz
```
Then, configure Intel HEXL with `-DHEXL_DOCS=ON` (see [Compile-time options](#compile-time-options)).
 To build Doxygen documentation, after configuring Intel HEXL with `-DHEXL_DOCS=ON`, run
```
cmake --build build --target docs
```
To view the generated Doxygen documentation, open the generated `docs/doxygen/html/index.html` file in a web browser.

# Contributing
This project welcomes external contributions. To contribute to Intel HEXL, see [CONTRIBUTING.md](CONTRIBUTING.md).
We encourage feedback and suggestions via [Github Issues](https://github.com/intel/hexl/issues) as well as discussion via [Github Discussions](https://github.com/intel/hexl/discussions).

## Repository layout
Public headers reside in the `hexl/include` folder.
Private headers, e.g. those containing Intel(R) AVX-512 code should not be put in this folder.

# Citing Intel HEXL
To cite Intel HEXL, please use the following BibTeX entry.

### Version 1.2
```tex
    @misc{IntelHEXL,
        author={Boemer, Fabian and Kim, Sejun and Seifu, Gelila and de Souza, Fillipe DM and Gopal, Vinodh and others},
        title = {{I}ntel {HEXL} (release 1.2)},
        howpublished = {\url{https://github.com/intel/hexl}},
        month = september,
        year = 2021,
        key = {Intel HEXL}
    }
```

### Version 1.1
```tex
    @misc{IntelHEXL,
        author={Boemer, Fabian and Kim, Sejun and Seifu, Gelila and de Souza, Fillipe DM and Gopal, Vinodh and others},
        title = {{I}ntel {HEXL} (release 1.1)},
        howpublished = {\url{https://github.com/intel/hexl}},
        month = may,
        year = 2021,
        key = {Intel HEXL}
    }
```

### Version 1.0
```tex
    @misc{IntelHEXL,
        author={Boemer, Fabian and Kim, Sejun and Seifu, Gelila and de Souza, Fillipe DM and Gopal, Vinodh and others},
        title = {{I}ntel {HEXL} (release 1.0)},
        howpublished = {\url{https://github.com/intel/hexl}},
        month = april,
        year = 2021,
        key = {Intel HEXL}
    }
```

# Contributors
The Intel contributors to this project, sorted by last name, are
  - [Paky Abu-Alam](https://www.linkedin.com/in/paky-abu-alam-89797710/)
  - [Flavio Bergamaschi](https://www.linkedin.com/in/flavio-bergamaschi-1634141/)
  - [Fabian Boemer](https://www.linkedin.com/in/fabian-boemer-5a40a9102/) (lead)
  - [Jeremy Bottleson](https://www.linkedin.com/in/jeremy-bottleson-38852a7/)
  - [Jack Crawford](https://www.linkedin.com/in/jacklhcrawford/)
  - [Fillipe D.M. de Souza](https://www.linkedin.com/in/fillipe-d-m-de-souza-a8281820/)
  - [Sergey Ivanov](https://www.linkedin.com/in/sergey-ivanov-451b72195/)
  - [Akshaya Jagannadharao](https://www.linkedin.com/in/akshaya-jagannadharao/)
  - [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin-655735/)
  - [Sejun Kim](https://www.linkedin.com/in/sejun-kim-2b1b4866/)
  - [Nir Peled](https://www.linkedin.com/in/nir-peled-4a52266/)
  - [Kylan Race](https://www.linkedin.com/in/kylanrace/)
  - [Gelila Seifu](https://www.linkedin.com/in/gelila-seifu/)

In addition to the Intel contributors listed, we are also grateful to contributions to this project that are not reflected in the Git history:
  - [Antonis Papadimitriou](https://www.linkedin.com/in/apapadimitriou/)
