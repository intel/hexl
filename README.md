# Intel Homomorphic Encryption Acceleration Library (HEXL)
Intel:registered: HEXL is an open-source library which provides efficient implementations of integer arithmetic on Galois fields. Such arithmetic is prevalent in cryptography, particularly in homomorphic encryption (HE) schemes. Intel HEXL targets integer arithmetic with word-sized primes, typically 40-60 bits. Intel HEXL provides an API for 64-bit unsigned integers and targets Intel CPUs.

## Contents
- [Intel Homomorphic Encryption Acceleration Library (HEXL)](#intel-homomorphic-encryption-acceleration-library-hexl)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Building Intel HEXL](#building-intel-hexl)
    - [Dependencies](#dependencies)
    - [Compile-time options](#compile-time-options)
    - [Compiling Intel HEXL](#compiling-intel-hexl)
  - [Testing Intel HEXL](#testing-intel-hexl)
  - [Benchmarking Intel HEXL](#benchmarking-intel-hexl)
  - [Using Intel HEXL](#using-intel-hexl)
  - [Debugging](#debugging)
  - [Threading](#threading)
- [Community Adoption](#community-adoption)
- [Documentation](#documentation)
  - [Doxygen](#doxygen)
  - [Sphinx](#sphinx)
- [Contributing](#contributing)
  - [Repository layout](#repository-layout)
  - [Intel HEXL Publication](#intel-hexl-publication)
  - [Citing Intel HEXL](#citing-intel-hexl)
    - [Version 1.1.1](#version-111)

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

### Dependencies
We have tested Intel HEXL on the following operating systems:
- Ubuntu 18.04
- macOS 10.15
- Microsoft Windows 10

Intel HEXL requires the following dependencies:

| Dependency  | Version                                      |
|-------------|----------------------------------------------|
| CMake       | >= 3.5.1                                     |
| Compiler    | gcc >= 7.0, clang++ >= 5.0, MSVC >= 2019     |

For best performance, we recommend using a processor with AVX512-IFMA52 support, and a recent compiler (gcc >= 8.0, clang++ >= 6.0). To determine if your process supports AVX512-IFMA52, simply look for `HEXL_HAS_AVX512IFMA` during the configure step (see [Compiling Intel HEXL](#compiling-hexl)).


### Compile-time options
In addition to the standard CMake build options, Intel HEXL supports several compile-time flags to configure the build.
For convenience, they are listed below:

| CMake option                     | Values                 |                                                                          |
| ---------------------------------| ---------------------- | ------------------------------------------------------------------------ |
| HEXL_BENCHMARK                | ON / OFF (default ON)  | Set to ON to enable benchmark suite via Google benchmark                 |
| HEXL_COVERAGE                 | ON / OFF (default OFF) | Set to ON to enable coverage report of unit-tests                        |
| HEXL_DOCS                     | ON / OFF (default OFF) | Set to ON to enable building of documentation                            |
| HEXL_SHARED_LIB               | ON / OFF (default OFF) | Set to ON to enable building shared library                              |
| HEXL_TESTING                  | ON / OFF (default ON)  | Set to ON to enable building of unit-tests                               |
| HEXL_TREAT_WARNING_AS_ERROR   | ON / OFF (default OFF) | Set to ON to treat all warnings as error                                 |

### Compiling Intel HEXL
The instructions to build Intel HEXL are common between Linux, MacOS, and Windows.

To compile Intel HEXL from source code, first clone the repository into your current directory. Then, to configure the build, call
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

## Testing Intel HEXL
To run a set of unit tests via Googletest, configure and build Intel HEXL with `-DHEXL_TESTING=ON` (see [Compile-time options](#compile-time-options)).
Then, run
```bash
cmake --build build --target unittest
```
The unit-test executable itself is located at `build/test/unit-test`
## Benchmarking Intel HEXL
To run a set of benchmarks via Google benchmark, configure and build Intel HEXL with `-DHEXL_BENCHMARK=ON` (see [Compile-time options](#compile-time-options)).
Then, run
```bash
cmake --build build --target bench
```
The benchmark executable itself is located at `build/benchmark/bench_hexl`

## Using Intel HEXL
The `example` folder has an example of using Intel HEXL in a third-party project.

## Debugging
For optimal performance, Intel HEXL does not perform input validation. In many cases the time required for the validation would be longer than the execution of the function itself. To debug Intel HEXL, configure and build Intel HEXL with `-DCMAKE_BUILD_TYPE=Debug` (see [Compile-time options](#compile-time-options)). This will generate a debug version of the library, e.g. `libhexl.a`, that can be used to debug the execution.

**Note**, enabling `CMAKE_BUILD_TYPE=Debug` will result in a significant runtime overhead.
## Threading
Intel HEXL is single-threaded and thread-safe.

# Community Adoption

- Intel HEXL has been integrated to the following homomorphic encryption libraries:
- [Microsoft SEAL](https://github.com/microsoft/SEAL)
- [PALISADE](https://gitlab.com/palisade/palisade-release)

If you are aware of any other uses of Intel HEXL, please let us know!

# Documentation
See [https://intel.github.io/hexl](https://intel.github.io/hexl) for Doxygen documentation.

Intel HEXL supports documentation via Doxygen and sphinx.
To build documentation, first install `doxygen` and `graphviz`, e.g.
```bash
sudo apt-get install doxygen graphviz
```
Then, configure Intel HEXL with `-DHEXL_DOCS=ON` (see [Compile-time options](#compile-time-options)).
## Doxygen
 To build Doxygen documentation, after configuring Intel HEXL with `-DHEXL_DOCS=ON`, run
```
cmake --build build --target doxygen
```
To view the generated Doxygen documentation, open the generated `build/docs/doxygen/html/index.html` file in a web browser.

## Sphinx
To build the sphinx documentation, install `sphinx` and required dependencies `breathe, m2r2`, e.g.
```bash
sudo apt-get install python3-sphinx
pip3 install breathe m2r2
```

Then, after configuring Intel HEXL with `-DHEXL_DOCS=ON`, run
```bash
cmake --build build --target docs
```
To view the generated Sphinx documentation, open the generated `build/docs/sphinx/html/index.html` file in a web browser.

# Contributing

At this time, Intel HEXL does not accept external contributions. We encourage feedback and suggestions via issues.

For Intel developers, use [pre-commit](https://pre-commit.com/) to validate the formatting of the code.

Before contributing, please run
```bash
cmake --build build --target check unittest
```
and make sure pre-commit checks and all unit tests pass.

## Repository layout
Public headers reside in the `hexl/include` folder.
Private headers, e.g. those containing Intel(R) AVX-512 code should not be put in this folder.

## Intel HEXL Publication
For more details on Intel HEXL, see our whitepaper at https://arxiv.org/abs/2103.16400.

## Citing Intel HEXL
To cite Intel HEXL, please use the following BibTeX entry.

### Version 1.1.1
```tex
    @misc{IntelHEXL,
        title = {{I}ntel {HEXL} (release 1.1.1)},
        howpublished = {\url{https://arxiv.org/abs/2103.16400}},
        month = mar,
        year = 2021,
        key = {Intel HEXL}
    }
```
