#include "fann_test_data.h"

void FannTestData::SetUp() {
    FannTest::SetUp();

    numData = 2;
    numInput = 3;
    numOutput = 1;
    inputValue = 1.1;
    outputValue = 2.2;

    inputData = new fann_type *[numData];
    outputData = new fann_type *[numData];

    InitializeTrainDataStructure(numData, numInput, numOutput, inputValue, outputValue, inputData, outputData);
}

void FannTestData::TearDown() {
    FannTest::TearDown();
    delete(inputData);
    delete(outputData);
}

void FannTestData::InitializeTrainDataStructure(unsigned int numData,
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

void FannTestData::AssertTrainData(training_data &trainingData, unsigned int numData, unsigned int numInput,
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


TEST_F(FannTestData, CreateTrainDataFromPointerArrays) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);

    AssertTrainData(data, numData, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTestData, CreateTrainDataFromArrays) {
    fann_type input[] = {inputValue, inputValue, inputValue, inputValue, inputValue, inputValue};
    fann_type output[] = {outputValue, outputValue};
    data.set_train_data(numData, numInput, input, numOutput, output);

    AssertTrainData(data, numData, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTestData, CreateTrainDataFromCopy) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    training_data dataCopy(data);

    AssertTrainData(dataCopy, numData, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTestData, CreateTrainDataFromFile) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    data.save_train("tmpFile");
    training_data dataCopy;
    dataCopy.read_train_from_file("tmpFile");

    AssertTrainData(dataCopy, numData, numInput, numOutput, inputValue, outputValue);
}

void callBack(unsigned int pos, unsigned int numInput, unsigned int numOutput, fann_type *input, fann_type *output) {
    for(unsigned int i = 0; i < numInput; i++)
        input[i] = (fann_type) 1.2;
    for(unsigned int i = 0; i < numOutput; i++)
        output[i] = (fann_type) 2.3;
}

TEST_F(FannTestData, CreateTrainDataFromCallback) {
    data.create_train_from_callback(numData, numInput, numOutput, callBack);
    AssertTrainData(data, numData, numInput, numOutput, 1.2, 2.3);
}

TEST_F(FannTestData, ShuffleTrainData) {
    //only really ensures that the data doesn't get corrupted, a more complete test would need to check
    //that this was indeed a permutation of the original data
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    data.shuffle_train_data();
    AssertTrainData(data, numData, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTestData, MergeTrainData) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    training_data dataCopy(data);
    data.merge_train_data(dataCopy);
    AssertTrainData(data, numData*2, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTestData, SubsetTrainData) {
    data.set_train_data(numData, numInput, inputData, numOutput, outputData);
    //call merge 2 times to get 8 data samples
    data.merge_train_data(data);
    data.merge_train_data(data);

    data.subset_train_data(2, 5);

    AssertTrainData(data, 5, numInput, numOutput, inputValue, outputValue);
}

TEST_F(FannTestData, ScaleOutputData) {
    fann_type input[] = {0.0, 1.0, 0.5, 0.0, 1.0, 0.5};
    fann_type output[] = {0.0, 1.0};
    data.set_train_data(2, 3, input, 1, output);

    data.scale_output_train_data(-1.0, 2.0);

    EXPECT_DOUBLE_EQ(0.0, data.get_min_input());
    EXPECT_DOUBLE_EQ(1.0, data.get_max_input());
    EXPECT_DOUBLE_EQ(-1.0, data.get_min_output());
    EXPECT_DOUBLE_EQ(2.0, data.get_max_output());
}

TEST_F(FannTestData, ScaleInputData) {
    fann_type input[] = {0.0, 1.0, 0.5, 0.0, 1.0, 0.5};
    fann_type output[] = {0.0, 1.0};
    data.set_train_data(2, 3, input, 1, output);

    data.scale_input_train_data(-1.0, 2.0);
    EXPECT_DOUBLE_EQ(-1.0, data.get_min_input());
    EXPECT_DOUBLE_EQ(2.0, data.get_max_input());
    EXPECT_DOUBLE_EQ(0.0, data.get_min_output());
    EXPECT_DOUBLE_EQ(1.0, data.get_max_output());
}

TEST_F(FannTestData, ScaleData) {
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
    EXPECT_DOUBLE_EQ(2.0, data.get_train_output(1)[0]);

}

TEST_F(FannTestData, ScaleDataByANN) {
    // Input 0, input 1, and the output are scaled normally. Input 2
    // has a standard deviation of 0 and so is not scaled at all.
    fann_type input[] = {0.0, 1.0, 0.5, 1.0, 2.0, 0.5};
    fann_type output[] = {0.0, 1.5};
    data.set_train_data(2, 3, input, 1, output);

    neural_net net(LAYER, 2, 3, 1);
    net.set_scaling_params(data, -1.0, 1.0, 0.0, 1.0);
    net.scale_train(data);

    EXPECT_DOUBLE_EQ(-1.0, data.get_train_input(0)[0]);
    EXPECT_DOUBLE_EQ(-1.0, data.get_train_input(0)[1]);
    EXPECT_DOUBLE_EQ(0.5, data.get_train_input(0)[2]);
    EXPECT_DOUBLE_EQ(0.0, data.get_train_output(0)[0]);

    EXPECT_DOUBLE_EQ(1.0, data.get_train_input(1)[0]);
    EXPECT_DOUBLE_EQ(1.0, data.get_train_input(1)[1]);
    EXPECT_DOUBLE_EQ(0.5, data.get_train_input(1)[2]);
    EXPECT_DOUBLE_EQ(1.0, data.get_train_output(1)[0]);
}
