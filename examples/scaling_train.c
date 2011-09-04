#include "fann.h"

int main( int argc, char** argv )
{
	const unsigned int num_input = 3;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 4;
	const unsigned int num_neurons_hidden = 5;
	const float desired_error = (const float) 0.0001;
	const unsigned int max_epochs = 5000;
	const unsigned int epochs_between_reports = 1000;
	struct fann_train_data * data = NULL;
	struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_neurons_hidden, num_output);
	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);
	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
	data = fann_read_train_from_file( "scaling.data" );
	/* Just pass any param to perform scaling */
	if( argc > 1 )
	{
		fann_set_scaling_params(
		    ann,
			data,
			-1,	/* New input minimum */
			1,	/* New input maximum */
			-1,	/* New output minimum */
			1);	/* New output maximum */

		fann_scale_train( ann, data );

		/*		
		fann_save_train(data, "scaled_scaling.data");
		fann_descale_train( ann, data );
		fann_save_train(data, "descaled_scaling.data");
		*/
	}
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
	free( data );
	fann_save(ann, "scaling.net");
	fann_destroy(ann);
	return 0;
}
