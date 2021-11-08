# Example using Intel HEXL in an external application

To use Intel HEXL in an external application, you can use one of the three provided ways.

* Install Intel HEXL. Then, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(HEXL 1.2.3
    HINTS ${HEXL_HINT_DIR}
    REQUIRED)
target_link_libraries(<your target> HEXL::hexl)
```
If Intel HEXL is installed globally, `HEXL_HINT_DIR` is not needed. Otherwise, `HEXL_HINT_DIR` should be the directory containing  `HEXLConfig.cmake`, e.g. `${CMAKE_INSTALL_PREFIX}/lib/cmake/hexl-1.2.2/`

* Install Intel HEXL. Then, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
include(FindPkgConfig)
pkg_check_modules(HEXL REQUIRED IMPORTED_TARGET hexl)
target_link_libraries(<your target> PkgConfig::HEXL)
```

* Install Intel HEXL from vcpkg. Then, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(HEXL CONFIG REQUIRED)
target_link_libraries(<your target> HEXL::hexl)
```
