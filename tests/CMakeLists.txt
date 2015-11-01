project (fann_tests)
include(CheckCXXCompilerFlag)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib/googletest/include)

CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

ADD_EXECUTABLE(fann_tests main.cpp fann_test.cpp fann_test_data.cpp fann_test_train.cpp)
target_link_libraries(fann_tests gtest doublefann)