# Example using Intel HE Acceleration Library in an external application

To use Intel HE Acceleration Library in an external application, you can use one of the three provided ways.

* Install Intel HE Acceleration Library. Then, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(HEXL 1.2.3
    HINTS ${HEXL_HINT_DIR}
    REQUIRED)
target_link_libraries(<your target> HEXL::hexl)
```
If Intel HE Acceleration Library is installed globally, `HEXL_HINT_DIR` is not needed. Otherwise, `HEXL_HINT_DIR` should be the directory containing  `HEXLConfig.cmake`, e.g. `${CMAKE_INSTALL_PREFIX}/lib/cmake/hexl-1.2.2/`

* Install Intel HE Acceleration Library. Then, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
include(FindPkgConfig)
pkg_check_modules(HEXL REQUIRED IMPORTED_TARGET hexl)
target_link_libraries(<your target> PkgConfig::HEXL)
```

* Install Intel HE Acceleration Library from vcpkg. Then, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(HEXL CONFIG REQUIRED)
target_link_libraries(<your target> HEXL::hexl)
```

To install a specific version of Intel HE Acceleration Library using vcpkg, use package versioning feature provided by vcpkg. An example manifest file is provided at `/path/to/hexl/example/vcpkg/vcpkg.json` with baseline pointing to a specific vcpkg commit. While building, overrides  will  force vcpkg to install and use the exact version specified.
