prefix=@PC_PREFIX@
libdir=@PC_LIB_DIR@
includedir=@PC_INCLUDE_DIR@

Name: fixedfann
Description: Fast Artificial Neural Network Library (fixed point version)
Version: @FANN_VERSION@
Libs: -L${libdir} -lfixedfann
Libs.private: -lm
Cflags: -I${includedir}
