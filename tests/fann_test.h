#ifndef FANN_FANN_TEST_H
#define FANN_FANN_TEST_H

#include "gtest/gtest.h"

#include "doublefann.h"
#include "fann_cpp.h"

using namespace FANN;

class FannTest : public testing::Test {
protected:
    neural_net net;
    training_data data;

    void AssertCreateAndCopy(neural_net &net, unsigned int numLayers, const unsigned int *layers, unsigned int neurons,
                             unsigned int connections);

    void AssertCreate(neural_net &net, unsigned int numLayers, const unsigned int *layers,
                      unsigned int neurons, unsigned int connections);

    void AssertWeights(neural_net &net, fann_type min, fann_type max, fann_type avg);

    virtual void SetUp();

    virtual void TearDown();
};

#endif
