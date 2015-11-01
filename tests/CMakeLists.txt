project (fann_tests)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib/googletest/include)

ADD_EXECUTABLE(fann_tests create_net.cpp main.cpp fann_test.h train_data.cpp fann_test.cpp)
target_link_libraries(fann_tests gtest doublefann)