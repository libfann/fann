include_directories(${CMAKE_SOURCE_DIR}/src/include)

add_definitions(-D_REENTRANT)

if(WIN32)
	set(M_LIBRARY) # Windows automatically link on the math standard library
else()
	find_library(M_LIBRARY
		NAMES m
		PATHS /usr/lib /usr/local/lib
		REQUIRED
	)
endif()


add_library(fann              SHARED floatfann.c)
add_library(floatfann         SHARED floatfann.c)
add_library(doublefann        SHARED doublefann.c)
add_library(fixedfann         SHARED fixedfann.c)

add_library(fann_static       STATIC floatfann.c)
add_library(floatfann_static  STATIC floatfann.c)
add_library(doublefann_static STATIC doublefann.c)
add_library(fixedfann_static  STATIC fixedfann.c)


foreach(p fann floatfann doublefann fixedfann)
	set_target_properties(${p} PROPERTIES
		VERSION ${FANN_VERSION}
		SOVERSION ${FANN_MAJOR_VERSION}
		DEFINE_SYMBOL FANN_DLL_EXPORTS
	)

	if(WIN32)
		set_target_properties(${p}_static PROPERTIES COMPILE_DEFINITIONS FANN_NO_DLL)
	endif()

	if(UNIX)
		# On unix-like, no need to add a suffix, since static and shared libraries use different extensions
		set_target_properties(${p}_static PROPERTIES OUTPUT_NAME ${p})
	endif()

	target_link_libraries(${p} ${M_LIBRARY})
	target_link_libraries(${p}_static ${M_LIBRARY})

endforeach()

set(TARGETS_TO_EXPORT fann floatfann doublefann fixedfann fann_static floatfann_static doublefann_static fixedfann_static)

install(TARGETS ${TARGETS_TO_EXPORT}
	EXPORT FANNTargets
	LIBRARY # Shared libraries on non-windows platforms
		DESTINATION ${INSTALL_DIR_LIB}
		COMPONENT runtimes
	RUNTIME # DLLs are runtimes on windows
		DESTINATION ${INSTALL_DIR_LIB}
		COMPONENT runtimes
	ARCHIVE # For .lib on windows and static libraries on other platforms
		DESTINATION ${INSTALL_DIR_ARCHIVE}
		COMPONENT development
)


########## Install and export #########

# Installs the headers
add_subdirectory(include)

# Generate a file that properly exports the targets for the BUILD tree
export(TARGETS ${TARGETS_TO_EXPORT}
	FILE "${PROJECT_BINARY_DIR}/FANNTargets.cmake"
)

# Generate a file that properly exports the targets for the INSTALL tree.
install(EXPORT FANNTargets
	DESTINATION "${INSTALL_DIR_CMAKE}"
	COMPONENT development
)

