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

    virtual void SetUp();

    virtual void TearDown();

};

#endif //FANN_FANN_TESTFIXTURE_H
