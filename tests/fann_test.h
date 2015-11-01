#ifndef FANN_FANN_TEST_H
#define FANN_FANN_TEST_H

#include "gtest/gtest.h"

#include "doublefann.h"
#include "fann_cpp.h"


class FannTest : public testing::Test {
protected:
    FANN::neural_net net;
    FANN::training_data data;

    void AssertCreateAndCopy(unsigned int numLayers, unsigned int *layers, unsigned int neurons,
                             unsigned int connections);

    void AssertCreate(FANN::neural_net &net, unsigned int numLayers, unsigned int *layers,
                      unsigned int neurons, unsigned int connections);

    void AssertWeights(fann_type min, fann_type max, fann_type avg);

    virtual void SetUp();

    virtual void TearDown();
};

#endif
