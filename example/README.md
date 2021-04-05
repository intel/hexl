# Example using Intel HEXL in an external application

To use Intel HEXL in an external application, first build Intel HEXL with `HEXL_EXPORT=ON`. Then, run `make install`.

Next, in your external application, add the following lines to your `CMakeLists.txt`:

```bash
find_package(HEXL 1.0.1
    HINTS ${HEXL_HINT_DIR}
    REQUIRED)
target_link_libraries(<your target> hexl)
```

If Intel HEXL is installed globally, `HEXL_HINT_DIR` is not needed. Otherwise, `HEXL_HINT_DIR` should be the directory containing  `HEXLConfig.cmake`, e.g. `${CMAKE_INSTALL_PREFIX}/lib/cmake/`
