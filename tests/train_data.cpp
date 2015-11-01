/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2012 Steffen Nissen (sn@leenissen.dk)

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

#include <stdio.h>
#include "gtest/gtest.h"

#include "doublefann.h"
#include "fann_cpp.h"
#include "fann_test.h"

TEST_F(FannTest, CreateTrainDataFromPointerArrays) {
    unsigned int num_data = 2;
    unsigned int num_input = 3;
    unsigned int num_output = 1;
    float input_value = 1.1f;
    float output_value = 2.2f;
    fann_type **input = new fann_type *[num_data];
    fann_type **output = new fann_type *[num_data];

    InitializeTrainDataStructure(num_data, num_input, num_output, input_value, output_value, input, output);

    data.set_train_data(num_data, num_input, input, num_output, output);

    AssertTrainData(num_data, num_input, num_output, input_value, output_value);
}

TEST_F(FannTest, CreateTrainDataFromArrays) {
    unsigned int num_data = 2;
    unsigned int num_input = 3;
    unsigned int num_output = 1;
    float input_value = 1.1f;
    float output_value = 2.2f;

    fann_type input[] = {input_value, input_value, input_value, input_value, input_value, input_value};
    fann_type output[] = {output_value, output_value};
    data.set_train_data(num_data, num_input, input, num_output, output);

    AssertTrainData(num_data, num_input, num_output, input_value, output_value);
}
