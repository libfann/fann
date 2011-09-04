/*
 *
 *  Fast Artificial Neural Network (fann) C++ Wrapper Sample
 *
 *  C++ wrapper XOR sample with functionality similar to xor_train.c
 *
 *  Copyright (C) 2004-2006 created by freegoldbar (at) yahoo dot com
 *
 *  This wrapper is free software; you can redistribute it and/or
 *  modify it under the terms of the GNU Lesser General Public
 *  License as published by the Free Software Foundation; either
 *  version 2.1 of the License, or (at your option) any later version.
 *
 *  This wrapper is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this library; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include "floatfann.h"
#include "fann_cpp.h"

#include <ios>
#include <iostream>
#include <iomanip>
using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::left;
using std::right;
using std::showpos;
using std::noshowpos;


// Callback function that simply prints the information to cout
int print_callback(FANN::neural_net &net, FANN::training_data &train,
    unsigned int max_epochs, unsigned int epochs_between_reports,
    float desired_error, unsigned int epochs, void *user_data)
{
    cout << "Epochs     " << setw(8) << epochs << ". "
         << "Current Error: " << left << net.get_MSE() << right << endl;
    return 0;
}

// Test function that demonstrates usage of the fann C++ wrapper
void xor_test()
{
    cout << endl << "XOR test started." << endl;

    const float learning_rate = 0.7f;
    const unsigned int num_layers = 3;
    const unsigned int num_input = 2;
    const unsigned int num_hidden = 3;
    const unsigned int num_output = 1;
    const float desired_error = 0.001f;
    const unsigned int max_iterations = 300000;
    const unsigned int iterations_between_reports = 1000;

    cout << endl << "Creating network." << endl;

    FANN::neural_net net;
    net.create_standard(num_layers, num_input, num_hidden, num_output);

    net.set_learning_rate(learning_rate);

    net.set_activation_steepness_hidden(1.0);
    net.set_activation_steepness_output(1.0);
    
    net.set_activation_function_hidden(FANN::SIGMOID_SYMMETRIC_STEPWISE);
    net.set_activation_function_output(FANN::SIGMOID_SYMMETRIC_STEPWISE);

    // Set additional properties such as the training algorithm
    //net.set_training_algorithm(FANN::TRAIN_QUICKPROP);

    // Output network type and parameters
    cout << endl << "Network Type                         :  ";
    switch (net.get_network_type())
    {
    case FANN::LAYER:
        cout << "LAYER" << endl;
        break;
    case FANN::SHORTCUT:
        cout << "SHORTCUT" << endl;
        break;
    default:
        cout << "UNKNOWN" << endl;
        break;
    }
    net.print_parameters();

    cout << endl << "Training network." << endl;

    FANN::training_data data;
    if (data.read_train_from_file("xor.data"))
    {
        // Initialize and train the network with the data
        net.init_weights(data);

        cout << "Max Epochs " << setw(8) << max_iterations << ". "
            << "Desired Error: " << left << desired_error << right << endl;
        net.set_callback(print_callback, NULL);
        net.train_on_data(data, max_iterations,
            iterations_between_reports, desired_error);

        cout << endl << "Testing network." << endl;

        for (unsigned int i = 0; i < data.length_train_data(); ++i)
        {
            // Run the network on the test data
            fann_type *calc_out = net.run(data.get_input()[i]);

            cout << "XOR test (" << showpos << data.get_input()[i][0] << ", " 
                 << data.get_input()[i][1] << ") -> " << *calc_out
                 << ", should be " << data.get_output()[i][0] << ", "
                 << "difference = " << noshowpos
                 << fann_abs(*calc_out - data.get_output()[i][0]) << endl;
        }
        
        cout << endl << "Saving network." << endl;

        // Save the network in floating point and fixed point
        net.save("xor_float.net");
        unsigned int decimal_point = net.save_to_fixed("xor_fixed.net");
        data.save_train_to_fixed("xor_fixed.data", decimal_point);

        cout << endl << "XOR test completed." << endl;
    }
}

/* Startup function. Syncronizes C and C++ output, calls the test function
   and reports any exceptions */
int main(int argc, char **argv)
{
    try
    {
        std::ios::sync_with_stdio(); // Syncronize cout and printf output
        xor_test();
    }
    catch (...)
    {
        cerr << endl << "Abnormal exception." << endl;
    }
    return 0;
}

/******************************************************************************/
