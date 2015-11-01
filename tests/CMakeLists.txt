project (fann_tests)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/include)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/lib/googletest/include)

ADD_EXECUTABLE(fann_tests create_net.cpp main.cpp fann_test.h train_data.cpp fann_test.cpp ../src/include/fann_data_cpp.h ../src/include/fann_training_data_cpp.h)
target_link_libraries(fann_tests gtest doublefann)