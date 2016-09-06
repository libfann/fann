IF(BIICODE)
	# Initializes block variables
	INIT_BIICODE_BLOCK()
	
	# Output folder for binaries
	SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../bin/fann/examples)
	SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG  ${CMAKE_CURRENT_SOURCE_DIR}/../bin/fann/examples)

	# Copy datasets for examples if exists
	if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/datasets)
	  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/datasets DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../bin/)
	ENDIF()

	# Include recipes block for CPP11 activation
	INCLUDE(biicode/cmake/tools)

	# Are examples present?
	LIST(FIND BII_BLOCK_EXES examples_parallel_train examples_present)
	SET(examples_present (NOT ${examples_present} EQUAL "-1")) # Depending on examples
	IF(${examples_present} AND NOT WIN32 AND NOT APPLE) # Linux doesn't have GetTickCount
		LIST(REMOVE_ITEM BII_BLOCK_EXES examples_parallel_train)		
	ENDIF()

	ADD_BIICODE_TARGETS()
	
	IF(${examples_present})
	   # This example needs CPP11
           ACTIVATE_CPP11(lasote_fann_examples_xor_sample)
	ENDIF()

	TARGET_COMPILE_OPTIONS(${BII_BLOCK_TARGET} INTERFACE -DGTEST_ENABLE_CATCH_EXCEPTIONS_=1)

	IF(MSVC)
	  TARGET_COMPILE_OPTIONS(${BII_LIB_TARGET} PUBLIC -DFANN_DLL_EXPORTS)
	ELSE()
	  IF(${examples_present})
	    TARGET_LINK_LIBRARIES(${BII_BLOCK_TARGET} INTERFACE gomp)
	  ENDIF()
	ENDIF()
ELSE()
cmake_minimum_required (VERSION 2.8)

if (NOT DEFINED CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif ()

project (FANN)

INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG(-std=c++11 COMPILER_SUPPORTS_CXX11)

IF(NOT COMPILER_SUPPORTS_CXX11)
  message(WARNING "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Tests will not be compiled. To enable tests use a compiler that supports C++11.")
ENDIF()

list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set (FANN_VERSION_MAJOR 2)
set (FANN_VERSION_MINOR 2)
set (FANN_VERSION_PATCH 0)
set (FANN_VERSION_STRING ${FANN_VERSION_MAJOR}.${FANN_VERSION_MINOR}.${FANN_VERSION_PATCH})

option(BUILD_SHARED_LIBS "build shared/static libs" ON)

INCLUDE(DefineInstallationPaths)


configure_file (cmake/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/src/include/config.h)
include_directories (${CMAKE_CURRENT_BINARY_DIR}/src/include/)

configure_file (cmake/fann.pc.cmake ${CMAKE_CURRENT_BINARY_DIR}/fann.pc @ONLY)

########### install files ###############

install (FILES ${CMAKE_CURRENT_BINARY_DIR}/fann.pc DESTINATION ${PKGCONFIG_INSTALL_DIR})

ADD_SUBDIRECTORY( src ) 

################# cpack ################

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Fast Artificial Neural Network Library (FANN)")
SET(CPACK_PACKAGE_VENDOR "Steffen Nissen")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.md")
SET(CPACK_PACKAGE_VERSION_MAJOR "${FANN_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${FANN_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${FANN_VERSION_PATCH}")
SET(CPACK_GENERATOR "TGZ;ZIP")
SET(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "Steffen Nissen") 
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
IF(WIN32 AND NOT UNIX)
  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
#  SET(CPACK_PACKAGE_ICON "${CMAKE_CURRENT_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
#  SET(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
#  SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} My Famous Project")
  SET(CPACK_NSIS_HELP_LINK "http:\\\\\\\\leenissen.dk/fann/")
  SET(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\leenissen.dk/fann/")
  SET(CPACK_NSIS_CONTACT "steffen.fann@gmail.com")
  SET(CPACK_NSIS_MODIFY_PATH ON)
ELSE(WIN32 AND NOT UNIX)
#  SET(CPACK_STRIP_FILES "bin/MyExecutable")
#  SET(CPACK_SOURCE_STRIP_FILES "")
ENDIF(WIN32 AND NOT UNIX)
#SET(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
INCLUDE(CPack)

################# config ################

set (FANN_USE_FILE ${CMAKE_CONFIG_DIR}/fann-use.cmake)
set (FANN_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
set (FANN_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include)
set (FANN_INCLUDE_DIRS ${FANN_INCLUDE_DIR})
set (FANN_LIBRARY_DIRS ${CMAKE_INSTALL_PREFIX}/lib)
set (FANN_LIBRARY fann)
set (FANN_LIBRARIES ${FANN_LIBRARY})
if (UNIX)
  list (APPEND FANN_LIBRARIES m)
endif ()

if (CMAKE_VERSION VERSION_LESS 2.8.8)
  configure_file (cmake/fann-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/fann-config.cmake @ONLY)
else ()

  include (CMakePackageConfigHelpers)

  configure_package_config_file (
    cmake/fann-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fann-config.cmake
    INSTALL_DESTINATION FANN_CMAKE_CONFIG_DIR
    PATH_VARS
     FANN_USE_FILE
     FANN_ROOT_DIR
     FANN_INCLUDE_DIR
     FANN_INCLUDE_DIRS
     FANN_LIBRARY_DIRS
   NO_CHECK_REQUIRED_COMPONENTS_MACRO
  )

endif ()

install (FILES
 ${CMAKE_CURRENT_BINARY_DIR}/fann-config.cmake
 cmake/fann-use.cmake
 DESTINATION ${CMAKE_CONFIG_DIR}
 )

################# compile tests ################

ADD_SUBDIRECTORY( lib/googletest )

if(COMPILER_SUPPORTS_CXX11)
  ADD_SUBDIRECTORY( tests )
endif()

ENDIF()
