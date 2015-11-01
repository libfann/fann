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

#include <stdio.h>
#include "gtest/gtest.h"

#include "doublefann.h"
#include "fann_cpp.h"
#include "fann_test.h"

TEST_F(FannTest, CreateTrainDataFromPointerArrays) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);

    AssertTrainData(data, numData, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTest, CreateTrainDataFromArrays) {
    fann_type input[] = {inputValue, inputValue, inputValue, inputValue, inputValue, inputValue};
    fann_type output[] = {outputValue, outputValue};
    data.set_train_data(numData, numInput, input, numOutput, output);

    AssertTrainData(data, numData, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTest, CreateTrainDataFromCopy) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    FANN::training_data dataCopy(data);

    AssertTrainData(dataCopy, numData, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTest, CreateTrainDataFromFile) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    data.save_train("tmpFile");
    FANN::training_data dataCopy;
    dataCopy.read_train_from_file("tmpFile");

    AssertTrainData(dataCopy, numData, numInput, numOutput, inputValue, outputValue);
}

void callBack(unsigned int pos, unsigned int numInput, unsigned int numOutput, fann_type *input, fann_type *output) {
    for(unsigned int i = 0; i < numInput; i++)
        input[i] = (fann_type) 1.2;
    for(unsigned int i = 0; i < numOutput; i++)
        output[i] = (fann_type) 2.3;
}

TEST_F(FannTest, CreateTrainDataFromCallback) {
    data.create_train_from_callback(numData, numInput, numOutput, callBack);
    AssertTrainData(data, numData, numInput, numOutput, 1.2, 2.3);
}

TEST_F(FannTest, ShuffleTrainData) {
    //only really ensures that the data doesn't get corrupted, a more complete test would need to check
    //that this was indeed a permutation of the original data
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    data.shuffle_train_data();
    AssertTrainData(data, numData, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTest, MergeTrainData) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    FANN::training_data dataCopy(data);
    data.merge_train_data(dataCopy);
    AssertTrainData(data, numData*2, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTest, SubsetTrainData) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    //call merge 2 times to get 8 data samples
    data.merge_train_data(data);
    data.merge_train_data(data);

    data.subset_train_data(2, 5);

    AssertTrainData(data, 5, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTest, ScaleOutputData) {
    fann_type input[] = {0.0, 1.0, 0.5, 0.0, 1.0, 0.5};
    fann_type output[] = {0.0, 1.0};
    data.set_train_data(2, 3, input, 1, output);

    data.scale_output_train_data(-1.0, 2.0);

    EXPECT_DOUBLE_EQ(0.0, data.get_min_input());
    EXPECT_DOUBLE_EQ(1.0, data.get_max_input());
    EXPECT_DOUBLE_EQ(-1.0, data.get_min_output());
    EXPECT_DOUBLE_EQ(2.0, data.get_max_output());
}

TEST_F(FannTest, ScaleInputData) {
    fann_type input[] = {0.0, 1.0, 0.5, 0.0, 1.0, 0.5};
    fann_type output[] = {0.0, 1.0};
    data.set_train_data(2, 3, input, 1, output);

    data.scale_input_train_data(-1.0, 2.0);
    EXPECT_DOUBLE_EQ(-1.0, data.get_min_input());
    EXPECT_DOUBLE_EQ(2.0, data.get_max_input());
    EXPECT_DOUBLE_EQ(0.0, data.get_min_output());
    EXPECT_DOUBLE_EQ(1.0, data.get_max_output());
}

TEST_F(FannTest, ScaleData) {
    fann_type input[] = {0.0, 1.0, 0.5, 0.0, 1.0, 0.5};
    fann_type output[] = {0.0, 1.0};
    data.set_train_data(2, 3, input, 1, output);

    data.scale_train_data(-1.0, 2.0);

    for(unsigned int i = 0; i < 2; i++) {
        fann_type *train_input = data.get_train_input(i);
        EXPECT_DOUBLE_EQ(-1.0, train_input[0]);
        EXPECT_DOUBLE_EQ(2.0, train_input[1]);
        EXPECT_DOUBLE_EQ(0.5, train_input[2]);
    }

    EXPECT_DOUBLE_EQ(-1.0, data.get_train_output(0)[0]);
    EXPECT_DOUBLE_EQ(2.0, data.get_train_output(0)[1]);

}
