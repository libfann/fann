#include <floatfann.h>
#include <stdlib.h>
#include <time.h>

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
	GLfloat *data;

	if (argc < 2)
		return -1;

	i = atoi(argv[1]);

	ann = fann_create_standard(5, i, i, i, i, i);
	fann_set_activation_function_hidden(ann, FANN_LINEAR_PIECE_LEAKY);
	fann_set_activation_function_output(ann, FANN_SIGMOID);
	input = calloc(sizeof(fann_type), ann->num_input);
	desired_output = calloc(sizeof(fann_type), ann->num_output);

	srand(time(NULL));

	for (i = 0; i < ann->num_input; i++)
		input[i] = ((float)rand()/RAND_MAX)-0.5;

	for (i = 0; i < ann->num_output; i++)
		desired_output[i] = ((float)rand()/RAND_MAX)-0.5;

	fann_print_parameters(ann);

	for (i = 0; i < 10; i++) {
		fanntest(ann, input, output, desired_output, 1);
		fanntest(ann, input, output, desired_output, 0);
	}

	return 0;
}

