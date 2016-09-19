#ifndef FANN_FANN_TEST_DATA_H
#define FANN_FANN_TEST_DATA_H

#include "gtest/gtest.h"

#include "doublefann.h"
#include "fann_cpp.h"
#include "fann_test.h"

class FannTestData : public FannTest {
protected:
    unsigned int numData;
    unsigned int numInput;
    unsigned int numOutput;
    fann_type inputValue;
    fann_type outputValue;

    fann_type **inputData;
    fann_type **outputData;

    virtual void SetUp();

    virtual void TearDown();

    void AssertTrainData(FANN::training_data &trainingData, unsigned int numData, unsigned int numInput,
                         unsigned int numOutput, fann_type inputValue, fann_type outputValue);


    void InitializeTrainDataStructure(unsigned int numData, unsigned int numInput, unsigned int numOutput,
                                      fann_type inputValue, fann_type outputValue, fann_type **inputData,
                                      fann_type **outputData);
};

#endif
