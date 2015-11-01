/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

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

TEST_F(FannTest, CreateStandardThreeLayers) {
    ASSERT_TRUE(net.create_standard(3, 2, 3, 4));
	unsigned int layers[] = {2, 3, 4};
    AssertCreateAndCopy(3, layers, 11, 25);
}

TEST_F(FannTest, CreateStandardFourLayers) {
    ASSERT_TRUE(net.create_standard(4, 2, 3, 4, 5));
	unsigned int layers[] = {2, 3, 4, 5};
    AssertCreateAndCopy(4, layers, 17, 50);
}

TEST_F(FannTest, CreateStandardFourLayersArray) {
	unsigned int layers[] = {2, 3, 4, 5};
	ASSERT_TRUE(net.create_standard_array(4, layers));
    AssertCreateAndCopy(4, layers, 17, 50);
}

TEST_F(FannTest, CreateSparseFourLayers) {
	ASSERT_TRUE(net.create_sparse(0.5f, 4, 2, 3, 4, 5));
	unsigned int layers[] = {2, 3, 4, 5};
    AssertCreateAndCopy(4, layers, 17, 31);
}

TEST_F(FannTest, CreateSparseArrayFourLayers) {
	unsigned int layers[] = {2, 3, 4, 5};
	ASSERT_TRUE(net.create_sparse_array(0.5f, 4, layers));
    AssertCreateAndCopy(4, layers, 17, 31);
}

TEST_F(FannTest, CreateSparseArrayWithMinimalConnectivity) {
	unsigned int layers[] = {2, 2, 2};
	ASSERT_TRUE(net.create_sparse_array(0.01f, 3, layers));
    AssertCreateAndCopy(3, layers, 8, 8);
}

TEST_F(FannTest, CreateShortcutFourLayers) {
	ASSERT_TRUE(net.create_shortcut(4, 2, 3, 4, 5));
	unsigned int layers[] = {2, 3, 4, 5};
    AssertCreateAndCopy(4, layers, 15, 83);
	EXPECT_EQ(FANN::SHORTCUT, net.get_network_type());
}

TEST_F(FannTest, CreateShortcutArrayFourLayers) {
	unsigned int layers[] = {2, 3, 4, 5};
	ASSERT_TRUE(net.create_shortcut_array(4, layers));
    AssertCreateAndCopy(4, layers, 15, 83);
	EXPECT_EQ(FANN::SHORTCUT, net.get_network_type());
}

TEST_F(FannTest, CreateFromFile) {
    ASSERT_TRUE(net.create_standard(3, 2, 3, 4));
    ASSERT_TRUE(net.save("tmpfile"));
    net.destroy();
    ASSERT_TRUE(net.create_from_file("tmpfile"));

    unsigned int layers[] = {2, 3, 4};
    AssertCreateAndCopy(3, layers, 11, 25);
}

TEST_F(FannTest, RandomizeWeights) {
    net.create_standard(2, 20, 10);
    net.randomize_weights(-1.0, 1.0);
    AssertWeights(-1.0, 1.0, 0);
}
