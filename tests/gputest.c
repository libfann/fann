#include <floatfann.h>

void
fanntest(struct fann *ann, fann_type *input, fann_type *output, fann_type *desired_output, int gl)
{
	double a, b;
	struct timeval now;
	int o;

	ann->gl = gl;

	gettimeofday(&now, NULL);
	b = now.tv_sec * 1000000;
	b += now.tv_usec;

	fann_reset_MSE(ann);
	fann_train(ann, input, desired_output);
//	fann_run(ann, input);

	gettimeofday(&now, NULL);
	a = now.tv_sec * 1000000;
	a += now.tv_usec;

	fprintf(stderr, "%cPU: %f microseconds MSE: %0.10lf\n", gl? 'G': 'C', a - b, ann->MSE_value);
}

int
main(int argc, char **argv)
{
	fann_type *input;
	fann_type *output;
	fann_type *desired_output;
	struct fann *ann;
	int i;

	if (argc < 2)
		return -1;

	i = atoi(argv[1]);

	ann = fann_create_standard(5, i, i, i, i, i);
	fann_set_activation_function_hidden(ann, FANN_LINEAR_PIECE_LEAKY);
	fann_set_activation_function_output(ann, FANN_SIGMOID);
	input = calloc(sizeof(fann_type), ann->num_input);
	desired_output = calloc(sizeof(fann_type), ann->num_output);

	for (i = 0; i < ann->num_output; i++)
		desired_output[i] = 0.73;

	fann_print_parameters(ann);

	for (i = 0; i < 10; i++)
		fanntest(ann, input, output, desired_output, 1);
	for (i = 0; i < 10; i++)
		fanntest(ann, input, output, desired_output, 0);

	return 0;
}

