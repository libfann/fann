#include "fann_test.h"

void FannTest::SetUp() {
    srand(0);
    fann_disable_seed_rand();
}

void FannTest::TearDown() {
    net.destroy();
    data.destroy_train();
}

void FannTest::AssertCreate(FANN::neural_net net, unsigned int num_layers, unsigned int *layers,
                            unsigned int neurons, unsigned int connections) {
    EXPECT_EQ(num_layers, net.get_num_layers());
    EXPECT_EQ(layers[0], net.get_num_input());
    EXPECT_EQ(layers[num_layers - 1], net.get_num_output());
    unsigned int *layers_res = new unsigned int[num_layers];
    net.get_layer_array(layers_res);
    for (unsigned int i = 0; i < num_layers; i++) {
        EXPECT_EQ(layers[i], layers_res[i]);
    }
    delete layers_res;

    EXPECT_EQ(neurons, net.get_total_neurons());
    EXPECT_EQ(connections, net.get_total_connections());

    AssertWeights(-0.09, 0.09, 0.0);
}

void FannTest::AssertCreateAndCopy(unsigned int num_layers, unsigned int *layers, unsigned int neurons,
                                   unsigned int connections) {
    AssertCreate(net, num_layers, layers, neurons, connections);
    FANN::neural_net net_copy(net);
    AssertCreate(net_copy, num_layers, layers, neurons, connections);
}

void FannTest::AssertWeights(fann_type expected_min_weight, fann_type expected_max_weight,
                             fann_type expected_avg_weight) {
    FANN::connection *connections = new FANN::connection[net.get_total_connections()];
    net.get_connection_array(connections);

    fann_type min_weight = connections[0].weight;
    fann_type max_weight = connections[0].weight;
    fann_type total_weight = 0.0;
    for (int i = 1; i < net.get_total_connections(); ++i) {
        if (connections[i].weight < min_weight)
            min_weight = connections[i].weight;
        if (connections[i].weight > max_weight)
            max_weight = connections[i].weight;
        total_weight += connections[i].weight;
    }

    EXPECT_NEAR(expected_min_weight, min_weight, 0.01);
    EXPECT_NEAR(expected_max_weight, max_weight, 0.01);
    EXPECT_NEAR(expected_avg_weight, total_weight / (fann_type) net.get_total_connections(), 0.1);
}


void FannTest::InitializeTrainDataStructure(unsigned int num_data,
                                            unsigned int num_input,
                                            unsigned int num_output,
                                            float input_value, float output_value,
                                            fann_type **input,
                                            fann_type **output) {
    for (unsigned int i = 0; i < num_data; i++) {
        input[i] = new fann_type[num_input];
        output[i] = new fann_type[num_output];
        for (unsigned int j = 0; j < num_input; j++)
            input[i][j] = input_value;
        for (unsigned int j = 0; j < num_output; j++)
            output[i][j] = output_value;
    }
}


void FannTest::AssertTrainData(unsigned int num_data, unsigned int num_input, unsigned int num_output,
                               fann_type input_value, fann_type output_value) {
    for (int i = 0; i < num_data; i++) {
        for (int j = 0; j < num_input; j++)
                EXPECT_EQ(input_value, this->data.get_input()[i][j]);
        for (int j = 0; j < num_output; j++)
                EXPECT_EQ(output_value, this->data.get_output()[i][j]);
    }
}

