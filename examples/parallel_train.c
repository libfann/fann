/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "fann.h"
#include "parallel_fann.h"

int main(int argc, const char* argv[])
{
	const unsigned int max_epochs = 1000;
	unsigned int num_threads = 1;
	struct fann_train_data *data;
	struct fann *ann;
	long before;
	float error;
	unsigned int i;

	if(argc == 2)
		num_threads = atoi(argv[1]);

	data = fann_read_train_from_file("../datasets/mushroom.train");
	ann = fann_create_standard(3, fann_num_input_train_data(data), 32, fann_num_output_train_data(data));

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID);

	before = GetTickCount();
	for(i = 1; i <= max_epochs; i++)
	{
		error = num_threads > 1 ? fann_train_epoch_irpropm_parallel(ann, data, num_threads) : fann_train_epoch(ann, data);
		printf("Epochs     %8d. Current error: %.10f\n", i, error);
	}
	printf("ticks %d", GetTickCount()-before);

	fann_destroy(ann);
	fann_destroy_train(data);

	return 0;
}
