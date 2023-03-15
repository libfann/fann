#include <fann.h>

int
main() {
	int i;
	struct fann *ann = fann_create_standard(3, 2, 5, 1);
	fann_type input[4][2] = {
		{ 0.0, 0.0 },
		{ 1.0, 0.0 },
		{ 0.0, 1.0 },
		{ 1.0, 1.0 }
	};
	fann_type output[4][1] = {
		{ 0.0 },
		{ 1.0 },
		{ 1.0 },
		{ 0.0 }
	};

	do {
		fann_reset_MSE(ann);
		for (i = 0; i < 4; i++)
			fann_train(ann, input[i], output[i]);
	} while (ann->MSE_value > 0.001);

	fprintf(stderr, "MSE: %f\n", ann->MSE_value);
}
