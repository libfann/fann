#ifndef FANN_FANN_TESTFIXTURE_H
#define FANN_FANN_TESTFIXTURE_H

#include "gtest/gtest.h"

#include "doublefann.h"
#include "fann_cpp.h"


class FannTest : public testing::Test {
protected:
    FANN::neural_net net;
    FANN::training_data data;

    unsigned int numData;
    unsigned int numInput;
    unsigned int numOutput;
    fann_type inputValue;
    fann_type outputValue;

    fann_type **inputData;
    fann_type **outputData;

    void AssertCreateAndCopy(unsigned int numLayers, unsigned int *layers, unsigned int neurons,
                             unsigned int connections);

    void AssertCreate(FANN::neural_net &net, unsigned int numLayers, unsigned int *layers,
                      unsigned int neurons, unsigned int connections);

    void AssertWeights(fann_type min, fann_type max, fann_type avg);

    void AssertTrainData(FANN::training_data &trainingData, unsigned int numData, unsigned int numInput,
                         unsigned int numOutput, fann_type inputValue, fann_type outputValue);

    virtual void SetUp();

    virtual void TearDown();

    void InitializeTrainDataStructure(unsigned int numData, unsigned int numInput, unsigned int numOutput,
                                      fann_type inputValue, fann_type outputValue, fann_type **inputData,
                                      fann_type **outputData);
};

#endif //FANN_FANN_TESTFIXTURE_H
