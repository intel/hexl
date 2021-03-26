# Example using Intel HEXL in an external application

To use Intel HEXL in an external application, first build Intel HEXL with `HEXL_EXPORT=ON`. Then, run `make install`.

Next, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(IntelHEXL 1.0.0
    HINTS ${INTEL_HEXL_HINT_DIR}
    REQUIRED)
target_link_libraries(<your target> intel_hexl)
```

If Intel HEXL is installed globally, `INTEL_HEXL_HINT_DIR` is not needed. Otherwise, `INTEL_HEXL_HINT_DIR` should be the directory containing  `IntelHEXLConfig.cmake`, e.g. `${CMAKE_INSTALL_PREFIX}/lib/cmake/`
