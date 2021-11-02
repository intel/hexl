# This library only supports "x64" architecture
vcpkg_fail_port_install(ON_ARCH "x86" "arm" "arm64")

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO intel/hexl
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release)

if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    set(HEXL_SHARED OFF)
else()
    set(HEXL_SHARED ON)
endif()

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    DISABLE_PARALLEL_CONFIGURE
    OPTIONS
        "-DHEXL_BENCHMARK=OFF"
        "-DHEXL_COVERAGE=OFF"
        "-DHEXL_TESTING=OFF"
        "-DHEXL_SHARED_LIB=${HEXL_SHARED}"
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME "HEXL" CONFIG_PATH "lib/cmake/hexl-1.2.3")

vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")

file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}" RENAME "copyright")

vcpkg_copy_pdbs()
