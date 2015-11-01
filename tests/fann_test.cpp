#include "fann_test.h"

void FannTest::SetUp() {
    //ensure random generator is seeded at a known value to ensure reproducible results
    srand(0);
    fann_disable_seed_rand();

    numData = 2;
    numInput = 3;
    numOutput = 1;
    inputValue = 1.1;
    outputValue = 2.2;

    inputData = new fann_type *[numData];
    outputData = new fann_type *[numData];

    InitializeTrainDataStructure(numData, numInput, numOutput, inputValue, outputValue, inputData, outputData);
}

void FannTest::TearDown() {
    delete(inputData);
    delete(outputData);
    net.destroy();
    data.destroy_train();
}

void FannTest::AssertCreate(FANN::neural_net &net, unsigned int numLayers, unsigned int *layers,
                            unsigned int neurons, unsigned int connections) {
    EXPECT_EQ(numLayers, net.get_num_layers());
    EXPECT_EQ(layers[0], net.get_num_input());
    EXPECT_EQ(layers[numLayers - 1], net.get_num_output());
    unsigned int *layers_res = new unsigned int[numLayers];
    net.get_layer_array(layers_res);
    for (unsigned int i = 0; i < numLayers; i++) {
        EXPECT_EQ(layers[i], layers_res[i]);
    }
    delete layers_res;

    EXPECT_EQ(neurons, net.get_total_neurons());
    EXPECT_EQ(connections, net.get_total_connections());

    AssertWeights(-0.09, 0.09, 0.0);
}

void FannTest::AssertCreateAndCopy(unsigned int numLayers, unsigned int *layers, unsigned int neurons,
                                   unsigned int connections) {
    AssertCreate(net, numLayers, layers, neurons, connections);
    FANN::neural_net net_copy(net);
    AssertCreate(net_copy, numLayers, layers, neurons, connections);
}

void FannTest::AssertWeights(fann_type min, fann_type max,
                             fann_type avg) {
    FANN::connection *connections = new FANN::connection[net.get_total_connections()];
    net.get_connection_array(connections);

    fann_type minWeight = connections[0].weight;
    fann_type maxWeight = connections[0].weight;
    fann_type totalWeight = 0.0;
    for (int i = 1; i < net.get_total_connections(); ++i) {
        if (connections[i].weight < minWeight)
            minWeight = connections[i].weight;
        if (connections[i].weight > maxWeight)
            maxWeight = connections[i].weight;
        totalWeight += connections[i].weight;
    }

    EXPECT_NEAR(min, minWeight, 0.01);
    EXPECT_NEAR(max, maxWeight, 0.01);
    EXPECT_NEAR(avg, totalWeight / (fann_type) net.get_total_connections(), 0.1);
}


void FannTest::InitializeTrainDataStructure(unsigned int numData,
                                            unsigned int numInput,
                                            unsigned int numOutput,
                                            fann_type inputValue, fann_type outputValue,
                                            fann_type **inputData,
                                            fann_type **outputData) {
    for (unsigned int i = 0; i < numData; i++) {
        inputData[i] = new fann_type[numInput];
        outputData[i] = new fann_type[numOutput];
        for (unsigned int j = 0; j < numInput; j++)
            inputData[i][j] = inputValue;
        for (unsigned int j = 0; j < numOutput; j++)
            outputData[i][j] = outputValue;
    }
}


void FannTest::AssertTrainData(FANN::training_data &trainingData, unsigned int numData, unsigned int numInput,
                               unsigned int numOutput, fann_type inputValue, fann_type outputValue) {
    EXPECT_EQ(numData, trainingData.length_train_data());
    EXPECT_EQ(numInput, trainingData.num_input_train_data());
    EXPECT_EQ(numOutput, trainingData.num_output_train_data());

    for (int i = 0; i < numData; i++) {
        for (int j = 0; j < numInput; j++)
                EXPECT_DOUBLE_EQ(inputValue, trainingData.get_input()[i][j]);
        for (int j = 0; j < numOutput; j++)
                EXPECT_DOUBLE_EQ(outputValue, trainingData.get_output()[i][j]);
    }
}

