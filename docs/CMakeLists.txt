# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Build Doxygen documentation
find_package(Doxygen REQUIRED)

set(DOXYGEN_INDEX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/doxygen/xml/indexl.html)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doxygen)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)

# Create Doxyfile
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) # Doxygen won't create this for us

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   DEPENDS ${HEXL_INC_ROOT_DIR}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating Doxygen documentation")

add_custom_target(doxygen DEPENDS ${DOXYGEN_INDEX_FILE})

# Build Sphinx documentation

find_package(Sphinx REQUIRED)

set(SPHINX_CONF_IN ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in)
set(SPHINX_CONF_OUT ${CMAKE_CURRENT_SOURCE_DIR}/conf.py)
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_SOURCE_DIR}/sphinx)

# Create Sphinx config
configure_file(${SPHINX_CONF_IN} ${SPHINX_CONF_OUT} @ONLY)
add_custom_target(sphinx
                  COMMAND ${SPHINX_EXECUTABLE} -b html
                  # Tell Breathe where to find the Doxygen output
                  -Dbreathe_projects.IntelHEXL=${DOXYGEN_OUTPUT_DIR}/xml
                  ${SPHINX_SOURCE} ${SPHINX_BUILD}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  DEPENDS doxygen
                  ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                  ${DOXYGEN_INDEX_FILE}
                  COMMENT "Generating documentation with Sphinx")

add_custom_target(docs COMMAND make sphinx DEPENDS sphinx)
