prefix=@PC_PREFIX@
libdir=@PC_LIB_DIR@
includedir=@PC_INCLUDE_DIR@

Name: doublefann
Description: Fast Artificial Neural Network Library (double version)
Version: @FANN_VERSION@
Libs: -L${libdir} -ldoublefann
Libs.private: -lm
Cflags: -I${includedir}
