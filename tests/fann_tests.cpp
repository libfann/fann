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

#include "fann.h"
#include "fann_cpp.h"

// Create

void TestCreate(FANN::neural_net &net, unsigned int num_layers, unsigned int *layers, unsigned int neurons, unsigned int connections) 
{
	EXPECT_EQ(num_layers, net.get_num_layers());
	EXPECT_EQ(layers[0], net.get_num_input());
	EXPECT_EQ(layers[num_layers-1], net.get_num_output());
	unsigned int *layers_res = new unsigned int[num_layers];
	net.get_layer_array(layers_res);
	for (unsigned int i = 0; i < num_layers; i++)
	{
		EXPECT_EQ(layers[i], layers_res[i]);
	}
	delete layers_res;

	EXPECT_EQ(neurons, net.get_total_neurons());
	EXPECT_EQ(connections, net.get_total_connections());
}

TEST(Create, CreateStandardThreeLayers) {
    FANN::neural_net net;
    ASSERT_TRUE(net.create_standard(3, 2, 3, 4));
	unsigned int layers[] = {2, 3, 4};
	TestCreate(net, 3, layers, 11, 25);
}

TEST(Create, CreateStandardFourLayers) {
    FANN::neural_net net;
    ASSERT_TRUE(net.create_standard(4, 2, 3, 4, 5));
	unsigned int layers[] = {2, 3, 4, 5};
	TestCreate(net, 4, layers, 17, 50);
}

TEST(Create, CreateStandardWillFailIfLessParametersAreGiven) {
    FANN::neural_net net;
    EXPECT_FALSE(net.create_standard(3, 2, 3));
}

TEST(Create, CreateStandardWillFailIfLessParametersAreGivenForCLibrary) {
	EXPECT_EQ(NULL, fann_create_standard(3, 2, 3));
}

TEST(Create, CreateStandardFourLayersArray) {
	unsigned int layers[] = {2, 3, 4, 5};
	FANN::neural_net net;
	ASSERT_TRUE(net.create_standard_array(4, layers));
	TestCreate(net, 4, layers, 17, 50);
}

TEST(Create, CreateSparseFourLayers) {
	FANN::neural_net net;
	ASSERT_TRUE(net.create_sparse(0.5f, 4, 2, 3, 4, 5));
	unsigned int layers[] = {2, 3, 4, 5};
	TestCreate(net, 4, layers, 17, 31);
}

TEST(Create, CreateSparseArrayFourLayers) {
	FANN::neural_net net;
	unsigned int layers[] = {2, 3, 4, 5};
	ASSERT_TRUE(net.create_sparse_array(0.5f, 4, layers));
	TestCreate(net, 4, layers, 17, 31);
}

TEST(Create, CreateSparseArrayWithMinimalConnectivity) {
	FANN::neural_net net;
	unsigned int layers[] = {2, 2, 2};
	ASSERT_TRUE(net.create_sparse_array(0.01f, 3, layers));
	TestCreate(net, 3, layers, 8, 8);
}

TEST(Create, CreateSparseWillFailIfLessParametersAreGiven) {
	FANN::neural_net net;
	EXPECT_FALSE(net.create_sparse(0.5f, 3, 2, 3));
}

TEST(Create, CreateSparseWillFailIfLessParametersAreGivenForCLibrary) {
	EXPECT_EQ(NULL, fann_create_sparse(0.5f, 3, 2, 3));
}

TEST(Create, CreateShortcutFourLayers) {
	FANN::neural_net net;
	ASSERT_TRUE(net.create_shortcut(4, 2, 3, 4, 5));
	unsigned int layers[] = {2, 3, 4, 5};
	TestCreate(net, 4, layers, 15, 83);
	EXPECT_EQ(FANN_NETTYPE_SHORTCUT, net.get_network_type());
}

TEST(Create, CreateShortcutWillFailIfLessParametersAreGiven) {
	FANN::neural_net net;
	EXPECT_FALSE(net.create_shortcut(3, 2, 3));
}

TEST(Create, CreateShortcutWillFailIfLessParametersAreGivenForCLibrary) {
	EXPECT_EQ(NULL, fann_create_shortcut(3, 2, 3));
}

TEST(Create, CreateShortcutArrayFourLayers) {
	FANN::neural_net net;
	unsigned int layers[] = {2, 3, 4, 5};
	ASSERT_TRUE(net.create_shortcut_array(4, layers));
	TestCreate(net, 4, layers, 15, 83);
	EXPECT_EQ(FANN_NETTYPE_SHORTCUT, net.get_network_type());
}

TEST(CreateTrain, CreateTrainDataFromPointerArrays) {
	FANN::training_data data;
	fann_type **input = new fann_type*[2];
	fann_type **output = new fann_type*[2];
	for(int i = 0; i < 2; i++) {
		input[i] = new fann_type[3];
		output[i] = new fann_type[1];
		for(int j = 0; j < 3; j++) {
			input[i][j] = 1.1f;
		}
		output[i][0] = 2.2f;
	}

	data.set_train_data(2, 3, input, 1, output);

	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 3; j++) {
			EXPECT_EQ(1.1f, data.get_input()[i][j]);
		}
		EXPECT_EQ(2.2f, data.get_output()[i][0]);
	}
}

TEST(CreateTrain, CreateTrainDataFromArrays) {
	FANN::training_data data;
	fann_type input[] = {1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f};
	fann_type output[] = {2.2f, 2.2f};

	data.set_train_data(2, 3, input, 1, output);

	for(int i = 0; i < 2; i++) {
		for(int j = 0; j < 3; j++) {
			EXPECT_EQ(1.1f, data.get_input()[i][j]);
		}
		EXPECT_EQ(2.2f, data.get_output()[i][0]);
	}
}