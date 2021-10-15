# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_library(EASYLOGGINGPP_LIBRARY easyloggingpp)
find_path(EASYLOGGINGPP_INCLUDEDIR easylogging++.h PATH_SUFFIXES include)

find_package_handle_standard_args(EASYLOGGINGPP REQUIRED_VARS EASYLOGGINGPP_LIBRARY EASYLOGGINGPP_INCLUDEDIR)

if(EASYLOGGINGPP_FOUND)
    message(STATUS "easyloggingpp library found")
else()
    message(STATUS "easyloggingpp library not found, using EASYLOGGINGPP packaged with HEXL")
endif()

if(EASYLOGGINGPP_FOUND)
    add_library(easyloggingpp INTERFACE IMPORTED)
    target_link_libraries(easyloggingpp INTERFACE ${EASYLOGGINGPP_LIBRARY})
    target_include_directories(easyloggingpp INTERFACE ${EASYLOGGINGPP_INCLUDEDIR})
endif()
