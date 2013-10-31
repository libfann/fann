prefix=@PC_PREFIX@
libdir=@PC_LIB_DIR@
includedir=@PC_INCLUDE_DIR@

Name: flotfann
Description: Fast Artificial Neural Network Library (float version)
Version: @FANN_VERSION@
Libs: -L${libdir} -lfloatfann
Libs.private: -lm
Cflags: -I${includedir}
