#include "fann_test_train.h"

void FannTestTrain::SetUp() {
    FannTest::SetUp();
}

void FannTestTrain::TearDown() {
    FannTest::TearDown();

}

TEST_F(FannTestTrain, TrainSimpleXor) {
    net.create_standard(3, 2, 3, 1);

    data.set_train_data(4, 2, xorInput, 1, xorOutput);
    net.train_on_data(data, 100, 100, 0.001);

    EXPECT_LT(net.test_data(data), 0.001);
}
