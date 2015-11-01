/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2012 Steffen Nissen (sn@leenissen.dk)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include "gtest/gtest.h"

#include "doublefann.h"
#include "fann_cpp.h"
#include "fann_test.h"

TEST_F(FannTest, CreateTrainDataFromPointerArrays) {
    fann_type **input = new fann_type *[2];
    fann_type **output = new fann_type *[2];
    for (int i = 0; i < 2; i++) {
        input[i] = new fann_type[3];
        output[i] = new fann_type[1];
        for (int j = 0; j < 3; j++) {
            input[i][j] = 1.1f;
        }
        output[i][0] = 2.2f;
    }

    data.set_train_data(2, 3, input, 1, output);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(1.1f, data.get_input()[i][j]);
        }
        EXPECT_EQ(2.2f, data.get_output()[i][0]);
    }
}

TEST_F(FannTest, CreateTrainDataFromArrays) {
    fann_type input[] = {1.1f, 1.1f, 1.1f, 1.1f, 1.1f, 1.1f};
    fann_type output[] = {2.2f, 2.2f};

    data.set_train_data(2, 3, input, 1, output);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            EXPECT_EQ(1.1f, data.get_input()[i][j]);
        }
        EXPECT_EQ(2.2f, data.get_output()[i][0]);
    }
}