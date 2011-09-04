/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003 Steffen Nissen (lukesky@diku.dk)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA02111-1307USA
*/

#include <stdio.h>

#include "fann.h"

int main()
{
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 96;
	const float desired_error = (const float) 0.001;
	struct fann *ann;
	struct fann_train_data *train_data, *test_data;

	float momentum;

	train_data = fann_read_train_from_file("../benchmarks/datasets/robot.train");
	test_data = fann_read_train_from_file("../benchmarks/datasets/robot.test");

	for ( momentum = 0.0; momentum < 0.7; momentum += 0.1 )
	{
		printf("============= momentum = %f =============\n", momentum);

		ann = fann_create_standard(num_layers,
						train_data->num_input, num_neurons_hidden, train_data->num_output);

		fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);

		fann_set_learning_momentum(ann, momentum);

		fann_train_on_data(ann, train_data, 2000, 500, desired_error);

		printf("MSE error on train data: %f\n", fann_test_data(ann, train_data));
		printf("MSE error on test data : %f\n", fann_test_data(ann, test_data));

		fann_destroy(ann);
	}

	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	return 0;
}
