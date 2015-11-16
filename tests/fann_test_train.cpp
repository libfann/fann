#include "fann_test_train.h"

using namespace std;

void FannTestTrain::SetUp() {
    FannTest::SetUp();
}

void FannTestTrain::TearDown() {
    FannTest::TearDown();

}

TEST_F(FannTestTrain, TrainOnDateSimpleXor) {
    neural_net net(LAYER, 3, 2, 3, 1);

    data.set_train_data(4, 2, xorInput, 1, xorOutput);
    net.train_on_data(data, 100, 100, 0.001);

    EXPECT_LT(net.get_MSE(), 0.001);
    EXPECT_LT(net.test_data(data), 0.001);
}

TEST_F(FannTestTrain, TrainSimpleIncrementalXor) {
    neural_net net(LAYER, 3, 2, 3, 1);

    for(int i = 0; i < 100000; i++) {
        net.train((fann_type*) (const fann_type[]) {0.0, 0.0}, (fann_type*) (const fann_type[]) {0.0});
        net.train((fann_type*) (const fann_type[]) {1.0, 0.0}, (fann_type*) (const fann_type[]) {1.0});
        net.train((fann_type*) (const fann_type[]) {0.0, 1.0}, (fann_type*) (const fann_type[]) {1.0});
        net.train((fann_type*) (const fann_type[]) {1.0, 1.0}, (fann_type*) (const fann_type[]) {0.0});
    }

    EXPECT_LT(net.get_MSE(), 0.01);
}
