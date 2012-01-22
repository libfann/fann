PROJECT (FANN)
#SET(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 2.4)


SET(CMAKE_MODULE_PATH
	${CMAKE_SOURCE_DIR}/cmake/Modules 
)

SET(VERSION 2.1.99)

INCLUDE(DefineInstallationPaths)



configure_file( ${CMAKE_SOURCE_DIR}/cmake/config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/src/include/config.h )
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/src/include/)

configure_file( ${CMAKE_SOURCE_DIR}/cmake/fann.pc.cmake ${CMAKE_CURRENT_BINARY_DIR}/fann.pc @ONLY )

SET(PKGCONFIG_INSTALL_DIR /lib/pkgconfig)

########### install files ###############

INSTALL_FILES( ${PKGCONFIG_INSTALL_DIR} FILES  fann.pc )


ADD_SUBDIRECTORY( src ) 

