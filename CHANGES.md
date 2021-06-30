# Changes

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
