# Changes

## Version 1.2.1
- Fixes a bug in AVX512 floating-point implementation of element-wise vector-vector modular multiplication (https://github.com/microsoft/SEAL/issues/385)
- Fixes a bug in the NTT default allocator (https://gitlab.com/palisade/palisade-development/-/issues/323#note_662270512)
- Improves performance of EltwiseFMAModAVX512 on ICX (https://github.com/intel/hexl/pull/42)
- Improves performance of the native NTT
- Adds reference implementations for the radix-4 NTT

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
