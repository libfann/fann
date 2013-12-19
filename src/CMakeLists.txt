ADD_SUBDIRECTORY( include )

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/include)
ADD_DEFINITIONS(-D_REENTRANT)

IF(WIN32)
  ADD_DEFINITIONS(-DFANN_DLL_EXPORTS)
ENDIF(WIN32)

MACRO(FANN_TARGET cc_file libname)
  ADD_LIBRARY(${libname} SHARED ${cc_file})
  ADD_LIBRARY(${libname}_static STATIC ${cc_file})

  TARGET_LINK_LIBRARIES(${libname} m)
  TARGET_LINK_LIBRARIES(${libname}_static m)

  SET_TARGET_PROPERTIES(${libname} PROPERTIES VERSION ${VERSION} SOVERSION 2 )
  SET_TARGET_PROPERTIES(${libname}_static PROPERTIES VERSION ${VERSION} SOVERSION 2 )

  INSTALL(TARGETS ${libname} DESTINATION ${LIB_INSTALL_DIR} )
  INSTALL(TARGETS ${libname}_static DESTINATION ${LIB_INSTALL_DIR} )
ENDMACRO(FANN_TARGET cc_file libname)

########### targets ###############

FANN_TARGET(floatfann.c fann)
FANN_TARGET(floatfann.c floatfann)
FANN_TARGET(doublefann.c doublefann)
FANN_TARGET(fixedfann.c fixedfann)

