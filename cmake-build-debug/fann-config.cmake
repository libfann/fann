#                                               -*- cmake -*-
#
#  fann-config.cmake(.in)
#

# Use the following variables to compile and link against FANN:
#  FANN_FOUND              - True if FANN was found on your system
#  FANN_USE_FILE           - The file making FANN usable
#  FANN_DEFINITIONS        - Definitions needed to build with FANN
#  FANN_INCLUDE_DIR        - Directory where fann.h can be found
#  FANN_INCLUDE_DIRS       - List of directories of FANN and it's dependencies
#  FANN_LIBRARY            - FANN library location
#  FANN_LIBRARIES          - List of libraries to link against FANN library
#  FANN_LIBRARY_DIRS       - List of directories containing FANN' libraries
#  FANN_ROOT_DIR           - The base directory of FANN
#  FANN_VERSION_STRING     - A human-readable string containing the version
#  FANN_VERSION_MAJOR      - The major version of FANN
#  FANN_VERSION_MINOR      - The minor version of FANN
#  FANN_VERSION_PATCH      - The patch version of FANN


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was fann-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

####################################################################################

set ( FANN_FOUND 1 )
set ( FANN_USE_FILE     "/usr/local/lib/cmake/fann/fann-use.cmake" )

set ( FANN_DEFINITIONS  "" )
set ( FANN_INCLUDE_DIR  "/usr/local/include" )
set ( FANN_INCLUDE_DIRS "/usr/local/include" )
set ( FANN_LIBRARY      "fann" )
set ( FANN_LIBRARIES    "fann;m" )
set ( FANN_LIBRARY_DIRS "/usr/local/lib" )
set ( FANN_ROOT_DIR     "/usr/local" )

set ( FANN_VERSION_STRING "2.2.0" )
set ( FANN_VERSION_MAJOR  "2" )
set ( FANN_VERSION_MINOR  "2" )
set ( FANN_VERSION_PATCH  "0" )

