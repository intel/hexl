# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This will define the following variables:
#
#   HEXL_FOUND          - True if the system has the Intel HEXL library
#   HEXL_VERSION        - The full major.minor.patch version number
#   HEXL_VERSION_MAJOR  - The major version number
#   HEXL_VERSION_MINOR  - The minor version number
#   HEXL_VERSION_PATCH  - The patch version number


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was HEXLConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro)
find_package(CpuFeatures CONFIG)
if(NOT CpuFeatures_FOUND)
    message(WARNING "Could not find pre-installed CpuFeatures; using CpuFeatures packaged with HEXL")
endif()

include(${CMAKE_CURRENT_LIST_DIR}/HEXLTargets.cmake)

# Defines HEXL_FOUND: If Intel HEXL library was found
if(TARGET HEXL::hexl)
    set(HEXL_FOUND TRUE)
    message(STATUS "Intel HEXL found")
else()
    message(STATUS "Intel HEXL not found")
endif()

set(HEXL_VERSION "1.2.5")
set(HEXL_VERSION_MAJOR "1")
set(HEXL_VERSION_MINOR "2")
set(HEXL_VERSION_PATCH "5")

set(HEXL_DEBUG "OFF")
