ADD_SUBDIRECTORY( include ) 

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/include)
ADD_DEFINITIONS(-D_REENTRANT)
########### next target ###############

SET(floatfann_LIB_SRCS
floatfann.c
)

ADD_LIBRARY(floatfann SHARED ${floatfann_LIB_SRCS})

SET_TARGET_PROPERTIES(floatfann PROPERTIES VERSION ${VERSION} SOVERSION 2 )
INSTALL(TARGETS floatfann DESTINATION ${LIB_INSTALL_DIR} )


########### next target ###############

SET(doublefann_LIB_SRCS
doublefann.c
)

ADD_LIBRARY(doublefann SHARED ${doublefann_LIB_SRCS})

SET_TARGET_PROPERTIES(doublefann PROPERTIES VERSION ${VERSION} SOVERSION 2 )
INSTALL(TARGETS doublefann DESTINATION ${LIB_INSTALL_DIR} )


########### next target ###############

SET(fixedfann_LIB_SRCS
fixedfann.c
)

ADD_LIBRARY(fixedfann SHARED ${fixedfann_LIB_SRCS})

SET_TARGET_PROPERTIES(fixedfann PROPERTIES VERSION ${VERSION} SOVERSION 2 )
INSTALL(TARGETS fixedfann DESTINATION ${LIB_INSTALL_DIR} )


########### next target ###############

SET(fann_LIB_SRCS
floatfann.c
)

ADD_LIBRARY(fann SHARED ${fann_LIB_SRCS})

SET_TARGET_PROPERTIES(fann PROPERTIES VERSION ${VERSION} SOVERSION 2 )
INSTALL(TARGETS fann DESTINATION ${LIB_INSTALL_DIR} )

