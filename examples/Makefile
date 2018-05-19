# The makefile requires that the fann library has been built first and is either in the

GCC=gcc
CFLAGS=-I ../src/include
LDFLAGS=-L ../src/

TARGETS = xor_train xor_test xor_test_fixed simple_train steepness_train simple_test robot mushroom cascade_train scaling_test scaling_train
DEBUG_TARGETS = xor_train_debug xor_test_debug xor_test_fixed_debug cascade_train_debug

all: $(TARGETS)

%: %.c Makefile
	$(GCC) $(CFLAGS) $(LDFLAGS) -O3 $< -o $@ -lfann -lm

%_fixed: %.c Makefile
	$(GCC) $(CFLAGS) $(LDFLAGS) -O3 -DFIXEDFANN $< -o $@ -lfixedfann -lm

clean:
	rm -f $(TARGETS) $(DEBUG_TARGETS) xor_fixed.data *.net *~ *.obj *.exe *.tds noscale.txt withscale.txt scale_test_results.txt

runtest: $(TARGETS)
	@echo
	@echo Training network
	./xor_train

	@echo
	@echo Testing network with floats
	./xor_test

	@echo
	@echo Testing network with fixed points
	./xor_test_fixed

#below this line is only for debugging the fann library

rundebugtest: $(DEBUG_TARGETS)
	@echo
	@echo Training network
	./xor_train_debug

	@echo
	@echo Testing network with floats
	./xor_test_debug

	@echo
	@echo Testing network with fixed points
	./xor_test_fixed_debug

#compiletest is used to test whether the library will compile easily in other compilers
compiletest:
	gcc -O3 -ggdb -DDISABLE_PARALLEL_FANN -DDEBUG -Wall -Wformat-security -Wfloat-equal -Wshadow -Wpointer-arith -Wcast-qual -Wsign-compare -pedantic -std=c99 -I../src/ -I../src/include/ ../src/floatfann.c xor_train.c -o xor_train -lm
	gcc -O3 -ggdb -DDISABLE_PARALLEL_FANN -DDEBUG -Wall -Wformat-security -Wfloat-equal -Wshadow -Wpointer-arith -Wcast-qual -Wsign-compare -pedantic -std=c99 -DFIXEDFANN -I../src/ -I../src/include/ ../src/fixedfann.c xor_test.c -o xor_test -lm
	gcc -O3 -ggdb -DDISABLE_PARALLEL_FANN -DDEBUG -Wall -Wformat-security -Wfloat-equal -Wpointer-arith -Wcast-qual -Wsign-compare -pedantic -std=c99 -I../src/ -I../src/include/ ../src/floatfann.c xor_train.c -o xor_train -lm
	g++ -O3 -ggdb -DDISABLE_PARALLEL_FANN -DDEBUG -Wall -Wformat-security -Wfloat-equal -Wpointer-arith -Wcast-qual -Wsign-compare -pedantic -std=c++11 -I../src/ -I../src/include/ ../src/floatfann.c -lm xor_sample.cpp -o xor_train

quickcompiletest:
	gcc -O -ggdb -DDEBUG -Wall -Wformat-security -Wfloat-equal -Wshadow -Wpointer-arith -Wcast-qual -Wsign-compare -pedantic -std=c99 -I../src/ -I../src/include/ ../src/floatfann.c ../examples/xor_train.c -o ../examples/xor_train -lm

debug: $(DEBUG_TARGETS)

%_debug: %.c Makefile ../src/*c ../src/include/*h
	$(GCC) $(CFLAGS) $(LDFLAGS) -ggdb -DDEBUG -Wall -ansi -I../src/ ../src/floatfann.c $< -o $@ -lm

%_fixed_debug: %.c Makefile ../src/*c ../src/include/*h
	$(GCC) $(CFLAGS) $(LDFLAGS) -ggdb -DDEBUG -Wall -ansi -DFIXEDFANN -I../src/ ../src/fixedfann.c $< -o $@ -lm

rundebug: $(DEBUG_TARGETS)
	@echo
	@echo Training network
	valgrind --leak-check=yes --show-reachable=yes --leak-resolution=high --db-attach=yes ./xor_train_debug

	@echo
	@echo Testing network with floats
	valgrind --leak-check=yes --show-reachable=yes --leak-resolution=high --db-attach=yes ./xor_test_debug

	@echo
	@echo Testing network with fixed points
	valgrind --leak-check=yes --show-reachable=yes --leak-resolution=high --db-attach=yes ./xor_test_fixed_debug
