#ifndef FANN_FANN_TESTFIXTURE_H
#define FANN_FANN_TESTFIXTURE_H

#include "gtest/gtest.h"

#include "doublefann.h"
#include "fann_cpp.h"

//ensure random generator is seeded at a known value to ensure reproducible results

class FannTest : public testing::Test {
protected:
    FANN::neural_net net;
    FANN::training_data data;

    void AssertCreateAndCopy(unsigned int num_layers, unsigned int *layers, unsigned int neurons,
                             unsigned int connections);

    void AssertCreate(FANN::neural_net net, unsigned int num_layers, unsigned int *layers,
                          unsigned int neurons, unsigned int connections);

    void AssertWeights(fann_type expected_min_weight, fann_type expected_max_weight,
                           fann_type expected_avg_weight);

    void AssertTrainData(unsigned int num_data, unsigned int num_input, unsigned int num_output, fann_type input_value,
                         fann_type output_value);

    virtual void SetUp();

    virtual void TearDown();

    void InitializeTrainDataStructure(unsigned int num_data, unsigned int num_input, unsigned int num_output,
                                      float input_value, float output_value, fann_type **input,
                                      fann_type **output);
};

#endif //FANN_FANN_TESTFIXTURE_H
