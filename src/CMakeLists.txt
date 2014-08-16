ADD_SUBDIRECTORY( include ) 

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/include)
ADD_DEFINITIONS(-D_REENTRANT)
if (WIN32)
ADD_DEFINITIONS(-DFANN_DLL_EXPORTS)
endif (WIN32)
########### next target ###############

SET(floatfann_LIB_SRCS
floatfann.c
)

ADD_LIBRARY(floatfann ${floatfann_LIB_SRCS})

SET_TARGET_PROPERTIES(floatfann PROPERTIES VERSION ${VERSION} SOVERSION 2 )
INSTALL(TARGETS floatfann LIBRARY DESTINATION ${LIB_INSTALL_DIR}
                          ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
                          RUNTIME DESTINATION ${BIN_INSTALL_DIR} )


########### next target ###############

SET(doublefann_LIB_SRCS
doublefann.c
)

ADD_LIBRARY(doublefann ${doublefann_LIB_SRCS})

SET_TARGET_PROPERTIES(doublefann PROPERTIES VERSION ${VERSION} SOVERSION 2 )
INSTALL(TARGETS doublefann LIBRARY DESTINATION ${LIB_INSTALL_DIR}
                           ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
                           RUNTIME DESTINATION ${BIN_INSTALL_DIR} )


########### next target ###############

SET(fixedfann_LIB_SRCS
fixedfann.c
)

ADD_LIBRARY(fixedfann ${fixedfann_LIB_SRCS})

TARGET_LINK_LIBRARIES(fixedfann m)

SET_TARGET_PROPERTIES(fixedfann PROPERTIES VERSION ${VERSION} SOVERSION 2 )
INSTALL(TARGETS fixedfann LIBRARY DESTINATION ${LIB_INSTALL_DIR}
                          ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
                          RUNTIME DESTINATION ${BIN_INSTALL_DIR} )


########### next target ###############

SET(fann_LIB_SRCS
floatfann.c
)

ADD_LIBRARY(fann ${fann_LIB_SRCS})

TARGET_LINK_LIBRARIES(fann m)

SET_TARGET_PROPERTIES(fann PROPERTIES VERSION ${VERSION} SOVERSION 2 )
INSTALL(TARGETS fann LIBRARY DESTINATION ${LIB_INSTALL_DIR}
                     ARCHIVE DESTINATION ${LIB_INSTALL_DIR}
                     RUNTIME DESTINATION ${BIN_INSTALL_DIR} )

