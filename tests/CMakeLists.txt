project (fann_tests)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib/googletest/include)

ADD_EXECUTABLE(fann_tests fann_tests.cpp main.cpp)
target_link_libraries(fann_tests gtest doublefann)