########### install headers ###############

install(FILES
	fann.h
	floatfann.h
	doublefann.h
	fixedfann.h
	fann_activation.h
	fann_cascade.h
	fann_cpp.h
	fann_data.h
	fann_error.h
	fann_internal.h
	fann_io.h
	fann_train.h
	DESTINATION "${INSTALL_DIR_INCLUDE}"
	COMPONENT dev
)
