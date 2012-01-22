prefix=@CMAKE_INSTALL_PREFIX@
exec_prefix=@BIN_INSTALL_DIR@
libdir=@LIB_INSTALL_DIR@
includedir=@INCLUDE_INSTALL_DIR@
 
Name: fann
Description: Fast Artificial Neural Network Library
Version: @VERSION@
Libs: -L${libdir} -lm -lfann
Cflags: -I${includedir}
