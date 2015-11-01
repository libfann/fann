########### install files ###############

IF(NOT OPENMP_FOUND OR DISABLE_PARALLEL_FANN)
  SET(PARALLEL_INCLUDES "")
ELSE(NOT OPENMP_FOUND OR DISABLE_PARALLEL_FANN)
  SET(PARALLEL_INCLUDES parallel_fann.h parallel_fann.hpp)
ENDIF(NOT OPENMP_FOUND OR DISABLE_PARALLEL_FANN)

install (FILES fann.h doublefann.h fann_internal.h floatfann.h fann_data.h fixedfann.h fann_activation.h fann_cascade.h fann_error.h fann_train.h fann_io.h fann_cpp.h fann_data_cpp.h fann_training_data_cpp.h ${PARALLEL_INCLUDES} DESTINATION ${INCLUDE_INSTALL_DIR})

