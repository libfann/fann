#include "fann_test.h"

void FannTest::SetUp() {
    //ensure random generator is seeded at a known value to ensure reproducible results
    srand(0);
    fann_disable_seed_rand();
}

void FannTest::TearDown() {
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

TEST_F(FannTest, CreateStandardThreeLayers) {
    ASSERT_TRUE(net.create_standard(3, 2, 3, 4));
    unsigned int layers[] = {2, 3, 4};
    AssertCreateAndCopy(3, layers, 11, 25);
}

TEST_F(FannTest, CreateStandardFourLayers) {
    ASSERT_TRUE(net.create_standard(4, 2, 3, 4, 5));
    unsigned int layers[] = {2, 3, 4, 5};
    AssertCreateAndCopy(4, layers, 17, 50);
}

TEST_F(FannTest, CreateStandardFourLayersArray) {
    unsigned int layers[] = {2, 3, 4, 5};
    ASSERT_TRUE(net.create_standard_array(4, layers));
    AssertCreateAndCopy(4, layers, 17, 50);
}

TEST_F(FannTest, CreateSparseFourLayers) {
    ASSERT_TRUE(net.create_sparse(0.5f, 4, 2, 3, 4, 5));
    unsigned int layers[] = {2, 3, 4, 5};
    AssertCreateAndCopy(4, layers, 17, 31);
}

TEST_F(FannTest, CreateSparseArrayFourLayers) {
    unsigned int layers[] = {2, 3, 4, 5};
    ASSERT_TRUE(net.create_sparse_array(0.5f, 4, layers));
    AssertCreateAndCopy(4, layers, 17, 31);
}

TEST_F(FannTest, CreateSparseArrayWithMinimalConnectivity) {
    unsigned int layers[] = {2, 2, 2};
    ASSERT_TRUE(net.create_sparse_array(0.01f, 3, layers));
    AssertCreateAndCopy(3, layers, 8, 8);
}

TEST_F(FannTest, CreateShortcutFourLayers) {
    ASSERT_TRUE(net.create_shortcut(4, 2, 3, 4, 5));
    unsigned int layers[] = {2, 3, 4, 5};
    AssertCreateAndCopy(4, layers, 15, 83);
    EXPECT_EQ(FANN::SHORTCUT, net.get_network_type());
}

TEST_F(FannTest, CreateShortcutArrayFourLayers) {
    unsigned int layers[] = {2, 3, 4, 5};
    ASSERT_TRUE(net.create_shortcut_array(4, layers));
    AssertCreateAndCopy(4, layers, 15, 83);
    EXPECT_EQ(FANN::SHORTCUT, net.get_network_type());
}

TEST_F(FannTest, CreateFromFile) {
    ASSERT_TRUE(net.create_standard(3, 2, 3, 4));
    ASSERT_TRUE(net.save("tmpfile"));
    net.destroy();
    ASSERT_TRUE(net.create_from_file("tmpfile"));

    unsigned int layers[] = {2, 3, 4};
    AssertCreateAndCopy(3, layers, 11, 25);
}

TEST_F(FannTest, RandomizeWeights) {
    net.create_standard(2, 20, 10);
    net.randomize_weights(-1.0, 1.0);
    AssertWeights(-1.0, 1.0, 0);
}
