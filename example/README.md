# Example using Intel HE Acceleration Library in an external application

This directory provides an example program as an example of using Intel HE
Acceleration Library in an external application. There are three methods that
you can use to link Intel HE Acceleration Library to your external program.

Before selecting a method, first install Intel HE Acceleration Library following
the instructions provided in the [README.md](../README.md).

For all methods an example `CMakeLists.txt` is provided in the corresponding
directory for building the example program `example.cpp` provided in this
directory. Specifically, methods 1, 2, and 3 are found in the `cmake`,
`pkgconfig`, and `vcpkg` directories, respectively.

To run directly any of the provided methods use,
```bash
cmake -S <directory-for-method> -B build
cmake --build build -j
```
The compiled example program , `example`, can be found in the `build` directory.

Note that the Windows build of the example program may require some of the
flags used un the Intel HE Acceleration Library build in
[README.md](../README.md).

## Method 1 (cmake)

Once you have installed Intel HE Acceleration Library. Then, in your external
application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(HEXL 1.2.5
    HINTS ${HEXL_HINT_DIR}
    REQUIRED)
target_link_libraries(<your target> HEXL::hexl)
```
If Intel HE Acceleration Library is installed globally, `HEXL_HINT_DIR` is not
needed. Otherwise, `HEXL_HINT_DIR` should be the directory containing
`HEXLConfig.cmake`, e.g. `${CMAKE_INSTALL_PREFIX}/lib/cmake/hexl-1.2.5/`

## Method 2 (pkgconfig)

Once you have installed Intel HE Acceleration Library. Then, in your external
application, add the following lines to your `CMakeLists.txt`:

```bash
include(FindPkgConfig)
pkg_check_modules(HEXL REQUIRED IMPORTED_TARGET hexl)
target_link_libraries(<your target> PkgConfig::HEXL)
```

## Method 3 (vcpkg)

If you installed Intel HE Acceleration Library from vcpkg. Then, in your external
application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(HEXL CONFIG REQUIRED)
target_link_libraries(<your target> HEXL::hexl)
```

To install a specific version of Intel HE Acceleration Library using vcpkg, use the
package versioning feature provided by vcpkg. An example manifest file is
provided at `/path/to/hexl/example/vcpkg/vcpkg.json` with baseline pointing to
a specific vcpkg commit. While building, overrides  will  force vcpkg to
install and use the exact version specified.
