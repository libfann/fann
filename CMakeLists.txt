project(FANN)
#SET(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 2.8)

set(FANN_MAJOR_VERSION 2)
set(FANN_MINOR_VERSION 2)
set(FANN_PATCH_VERSION 0)
set(FANN_VERSION "${FANN_MAJOR_VERSION}.${FANN_MINOR_VERSION}.${FANN_PATCH_VERSION}")

configure_file("${CMAKE_SOURCE_DIR}/cmake/config.h.cmake" "${CMAKE_CURRENT_BINARY_DIR}/src/include/config.h")


########### install files ############

if(WIN32 AND NOT CYGWIN)
	set(INSTALL_DIR_LIB bin)    # .dll are in PATH, like executables
	set(INSTALL_DIR_CMAKE CMake)
else()
	set(INSTALL_DIR_LIB lib)
	set(INSTALL_DIR_CMAKE "${INSTALL_DIR_LIB}/cmake/FANN")
endif()

set(INSTALL_DIR_ARCHIVE lib)
set(INSTALL_DIR_INCLUDE include)
set(INSTALL_DIR_PC      share/pkgconfig)


########### pkgconfig file ############

foreach(p fann floatfann doublefann fixedfann)
	# Build tree
	set(PC_PREFIX "${CMAKE_BINARY_DIR}")
	set(PC_LIB_DIR "\${prefix}/src")
	set(PC_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/src/include")
	configure_file( "${CMAKE_SOURCE_DIR}/cmake/${p}.pc.cmake" "${CMAKE_CURRENT_BINARY_DIR}/${p}.pc" @ONLY )

	# Install tree
	set(PC_PREFIX "${CMAKE_INSTALL_PREFIX}")
	set(PC_LIB_DIR "\${prefix}/${INSTALL_DIR_LIB}")
	set(PC_INCLUDE_DIR "\${prefix}/${INSTALL_DIR_INCLUDE}")
	configure_file("${CMAKE_SOURCE_DIR}/cmake/${p}.pc.cmake" "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${p}.pc" @ONLY)
	install(FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${p}.pc" DESTINATION "${INSTALL_DIR_PC}")
endforeach()


########## Build the libraries #############

add_subdirectory(src)


########## Install and export ##########

# see http://www.cmake.org/Wiki/CMake/Tutorials/How_to_create_a_ProjectConfig.cmake_file

file(RELATIVE_PATH REL_INCLUDE_DIR "${CMAKE_INSTALL_PREFIX}/${INSTALL_DIR_CMAKE}" "${CMAKE_INSTALL_PREFIX}/${INSTALL_DIR_INCLUDE}")

# Write the file used to match a specific version of the library
configure_file(cmake/FANNConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/FANNConfigVersion.cmake" @ONLY)

# Write the file that defines de variables that reference the fann include dir and the fann libraries
## For the build tree
set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/src/include")
configure_file(cmake/FANNConfig.cmake.in "${PROJECT_BINARY_DIR}/FANNConfig.cmake" @ONLY)

## For the install tree
set(CONF_INCLUDE_DIRS "\${FANN_CMAKE_DIR}/${REL_INCLUDE_DIR}")
configure_file(cmake/FANNConfig.cmake.in "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FANNConfig.cmake" @ONLY)

install(
	FILES
		"${PROJECT_BINARY_DIR}/FANNConfigVersion.cmake"
		"${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FANNConfig.cmake"
	DESTINATION
		${INSTALL_DIR_CMAKE}
)

# Add the build tree of the project to the CMake user package registry 
# see http://www.cmake.org/cmake/help/v2.8.8/cmake.html#command:export
export(PACKAGE FANN)


################# cpack ################

set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Fast Artificial Neural Network Library (FANN)")
set(CPACK_PACKAGE_VENDOR "Steffen Nissen")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.txt")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${FANN_MAJOR_VERSION}")
set(CPACK_PACKAGE_VERSION_MINOR "${FANN_MINOR_VERSION}")
set(CPACK_PACKAGE_VERSION_PATCH "${FANN_PATCH_VERSION}")
set(CPACK_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Steffen Nissen") 
set(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
if(WIN32 AND NOT UNIX)
  # There is a bug in NSI that does not handle full unix paths properly. Make
  # sure there is at least one set of four (4) backlasshes.
#  SET(CPACK_PACKAGE_ICON "${CMake_SOURCE_DIR}/Utilities/Release\\\\InstallIcon.bmp")
#  SET(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MyExecutable.exe")
#  SET(CPACK_NSIS_DISPLAY_NAME "${CPACK_PACKAGE_INSTALL_DIRECTORY} My Famous Project")
  set(CPACK_NSIS_HELP_LINK "http:\\\\\\\\leenissen.dk/fann/")
  set(CPACK_NSIS_URL_INFO_ABOUT "http:\\\\\\\\leenissen.dk/fann/")
  set(CPACK_NSIS_CONTACT "sn@leenissen.dk")
  set(CPACK_NSIS_MODIFY_PATH ON)
else(WIN32 AND NOT UNIX)
#  SET(CPACK_STRIP_FILES "bin/MyExecutable")
#  SET(CPACK_SOURCE_STRIP_FILES "")
endif(WIN32 AND NOT UNIX)
#SET(CPACK_PACKAGE_EXECUTABLES "MyExecutable" "My Executable")
include(CPack)
