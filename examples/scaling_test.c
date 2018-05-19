#include <stdio.h>
#include "fann.h"

int main( int argc, char** argv )
{
	fann_type *calc_out;
	unsigned int i;
	int ret = 0;
	struct fann *ann;
	struct fann_train_data *data;
	printf("Creating network.\n");
	ann = fann_create_from_file("scaling.net");
	if(!ann)
	{
		printf("Error creating ann --- ABORTING.\n");
		return 0;
	}
	fann_print_connections(ann);
	fann_print_parameters(ann);
	printf("Testing network.\n");
	data = fann_read_train_from_file("../datasets/scaling.data");
	for(i = 0; i < fann_length_train_data(data); i++)
	{
		fann_reset_MSE(ann);
    	fann_scale_input( ann, data->input[i] );
		calc_out = fann_run( ann, data->input[i] );
		fann_descale_output( ann, calc_out );
		printf("Result %f original %f error %f\n",
			calc_out[0], data->output[i][0],
			(float) fann_abs(calc_out[0] - data->output[i][0]));
	}
	printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);
	return ret;
}
