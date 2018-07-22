/* This benchmark generates a random set of inputs and outputs
 * on a large network and tests the backpropagation speed in order
 * for the net to arrive at a certain level of error. */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#include "fann.h"
#include "parallel_fann.h"

//#define ONLY_TEST

#define NUM_INPUTS 60 //120 //300
#define NUM_OUTPUTS 10 //40 //100

int math_random(int low, int up) {
  fann_type r = rand() * (1.0 / (RAND_MAX + 1.0));
  r *= (up - low) + 1.0;
  return (int)r+low;
}

void gen_dataset(fann_type **inputs, fann_type **outputs, int setsize) {
    *inputs = calloc(1, sizeof(fann_type)*setsize*NUM_INPUTS);
    *outputs = calloc(1, sizeof(fann_type)*setsize*NUM_OUTPUTS);
    int ilen = NUM_INPUTS;
    int olen = NUM_OUTPUTS;
    int olen_1 = olen - 1;

    fann_type *in = *inputs;
    fann_type *out = *outputs;
    for (int j = 0; j < setsize; j++) {
        for (int k = 0; k < ilen; k++) in[k] = rand() & 1;
        //int r = rand() & olen_1;
        int r = math_random(0, olen_1);
        out[r] = 1;
        //printf("%d : %d\n", j, r);
        //for (int k = 0; k < olen; k++) {
        //    out[k] = (k == r) ? 1 : 0;
        //}
        in+= ilen;
        out+= olen;
    }
}

int main(int argc, char *argv[]) {
    const char fn_net[] = "benchmark_float.net";
	int setsize = 1000;
	unsigned int i = 0, ilen;
	unsigned int num_threads = 1;
	if(argc == 2)
		num_threads = atoi(argv[1]);
#ifndef ONLY_TEST
	fann_type *inputs, *outputs;
	gen_dataset(&inputs, &outputs, setsize);
#endif
	fann_type *test_inputs, *test_outputs;
	gen_dataset(&test_inputs, &test_outputs, setsize);

	struct fann_train_data *test_data;

#ifndef ONLY_TEST
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = NUM_INPUTS*2;
	const float desired_error = (const float) 0.0001;
	float desired_error_reached;
	const unsigned int max_epochs = 3000;
	const unsigned int epochs_between_reports = 10;
	struct fann *ann;
	struct fann_train_data *train_data;

	printf("Creating network.\n");

	train_data = fann_create_train_array(setsize, NUM_INPUTS, inputs, NUM_OUTPUTS, outputs);

	ann = fann_create_standard(num_layers,
					  train_data->num_input, num_neurons_hidden, train_data->num_output);

	printf("Training network.\n");

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID);
	//fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
	fann_set_learning_rate(ann, 0.5f);
	fann_randomize_weights(ann, -2.0f, 2.0f);

	/*fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL); */

	//fann_train_on_data(ann, train_data, max_epochs, epochs_between_reports, desired_error);


	long before = fann_mstime();
	for(i = 1; i <= max_epochs; i++)
	{
		long start = fann_mstime();
		double error = (num_threads > 1)
			? fann_train_epoch_irpropm_parallel(ann, train_data, num_threads) 
			: fann_train_epoch(ann, train_data);
		long elapsed = fann_mstime() - start;
		printf("Epochs     %8d. Current error: %.10f :: %ld\n", i, error, elapsed);
		desired_error_reached = fann_desired_error_reached(ann, desired_error);
		if(desired_error_reached == 0)
			break;
	}
	printf("Time spent %ld ms\n", fann_mstime()-before);

#else
	struct fann *ann = fann_create_from_file(fn_net);
#endif

	test_data = fann_create_train_array(setsize, NUM_INPUTS, test_inputs, NUM_OUTPUTS, test_outputs);
	ilen = fann_length_train_data(test_data);
	printf("Testing network. %d\n", ilen);


	fann_reset_MSE(ann);
	for(i = 0; i < ilen; i++)
	{
		fann_test(ann, test_data->input[i], test_data->output[i]);
	}

	printf("MSE error on test data: %f\n", fann_get_MSE(ann));

#ifndef ONLY_TEST
	printf("Saving network.\n");

	fann_save(ann, fn_net);
#endif

	printf("Cleaning up.\n");
	fann_destroy(ann);

#ifndef ONLY_TEST
	fann_destroy_train(train_data);
	free(inputs);
	free(outputs);
#endif
	fann_destroy_train(test_data);
	free(test_inputs);
	free(test_outputs);

	return 0;
}
