# Changes

## Version 1.2.3
- Fix to EltwiseReduceMod on AVX512-DQ processors (https://github.com/intel/hexl/issues/86)
- Update minimum CMake version to 3.13
- Fixes 3rd-party dependency commits (https://github.com/intel/hexl/pull/85)

## Version 1.2.2
- Fixes Barrett reduce native benchmark (https://github.com/intel/hexl/pull/65)
- Fixes 32 bit invntt (https://github.com/intel/hexl/pull/73)
- Fixes CMake 3.13 compilation (https://github.com/intel/hexl/pull/67)
- Added information of HElib integration to README (https://github.com/intel/hexl/pull/63)
- Added random number generator which generates a vector of specified size with random values drawn uniformly from [0, modulus) (https://github.com/intel/hexl/pull/62)
- Added random input values for benchmarks (https://github.com/intel/hexl/pull/66)
- Uses Generalized Barrett Reduction algorithm for EltwiseMultMod (https://github.com/intel/hexl/pull/68)
- Avoids memcpy operations on NTT (https://github.com/intel/hexl/pull/72)
- Added performance tips to README (https://github.com/intel/hexl/pull/74)
- Improves performance of Barrett Reduction (https://github.com/intel/hexl/pull/75)

## Version 1.2.1
- Fixes a bug in AVX512 floating-point implementation of element-wise vector-vector modular multiplication (https://github.com/microsoft/SEAL/issues/385)
- Fixes a bug in the NTT default constructor (https://gitlab.com/palisade/palisade-development/-/issues/329)
- Fixes a bug in the AVX512 NTT (https://github.com/intel/hexl/pull/58)
- Improves performance of EltwiseFMAModAVX512 on ICX (https://github.com/intel/hexl/pull/42)
- Improves performance of the native NTT
- Adds reference implementations for the radix-4 NTT
- Enables support for pre-built easylogging (https://github.com/intel/hexl/pull/57)

## Version 1.2.0
- Large performance improvement in large (N >= 16384) AVX512 NTTs via recursive implementations
- Slight performance improvements in AVX512IFMA NTT
- Slight performance improvements in element-wise modular multiplication
- Implement optimized AVX512DQ NTT for moduli < 32 bits
- Expands public API to include number theory, NTT twiddle factors
- Remove HEXL_DEBUG and HEXL_EXPORT options. The behavior now is to always export the cmake configuration files, and HEXL_DEBUG is enabled iff the CMAKE_BUILD_TYPE=Debug
- Added pkgconfig support

## Version 1.1.1
- Fix Google benchmark branch to point to "main" instead of "master"

## Version 1.1.0
- Added vector-vector and vector-scalar EltwiseSubMod
- Added vector-scalar version of EltwiseAddMod
- Added generic 128-bit division for Windows build
- Documentation hosted on a separate branch
- Enabled custom allocator for NTT class
- Fixed argument order in EltwiseReduceMod
- Fixed build warnings on Windows and Mac

## Version 1.0.1
- Removed intel- prefix from headers and library name
- Fixed CMake variables from 3rd party libraries from leaking
- Fixed warnings when HEXL_DEBUG=ON
