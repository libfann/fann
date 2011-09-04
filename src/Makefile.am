SUBDIRS = include

lib_LTLIBRARIES = libfloatfann.la libdoublefann.la libfixedfann.la libfann.la

AM_LDFLAGS = -version-info 2:1:0


AM_CFLAGS = -D_REENTRANT
if X86_64
AM_CFLAGS += -fPIC
endif
libfloatfann_la_SOURCES = floatfann.c
libdoublefann_la_SOURCES = doublefann.c
libfixedfann_la_SOURCES = fixedfann.c
libfann_la_SOURCES = floatfann.c

EXTRA_DIST = fann.c fann_io.c fann_train.c fann_train_data.c fann_error.c fann_cascade.c
