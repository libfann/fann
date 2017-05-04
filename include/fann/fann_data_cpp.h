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

#ifndef FANN_FANN_DATA_CPP_H_H
#define FANN_FANN_DATA_CPP_H_H

#include <stdarg.h>
#include <string>

/* Section: FANN C++ Datatypes
   This section includes enums and helper data types used by the two main classes <neural_net> and <training_data>
 */


/* Type: fann_type
   fann_type is the type used for the weights, inputs and outputs of the neural network.

	fann_type is defined as a:
	float - if you include fann.h or floatfann.h
	double - if you include doublefann.h
	int - if you include fixedfann.h (please be aware that fixed point usage is
			only to be used during execution, and not during training).
*/

namespace FANN {
    /* Enum: error_function_enum
	    Error function used during training.

	    ERRORFUNC_LINEAR - Standard linear error function.
	    ERRORFUNC_TANH - Tanh error function, usually better
		    but can require a lower learning rate. This error function aggressively targets outputs that
		    differ much from the desired, while not targeting outputs that only differ a little that much.
		    This activation function is not recommended for cascade training and incremental training.

	    See also:
		    <neural_net::set_train_error_function>, <neural_net::get_train_error_function>
    */
    enum error_function_enum {
        ERRORFUNC_LINEAR = FANN_ERRORFUNC_LINEAR,
        ERRORFUNC_TANH
    };

    /* Enum: stop_function_enum
	    Stop criteria used during training.

	    STOPFUNC_MSE - Stop criteria is Mean Square Error (MSE) value.
	    STOPFUNC_BIT - Stop criteria is number of bits that fail. The number of bits; means the
		    number of output neurons which differ more than the bit fail limit
		    (see <neural_net::get_bit_fail_limit>, <neural_net::set_bit_fail_limit>).
		    The bits are counted in all of the training data, so this number can be higher than
		    the number of training data.

	    See also:
		    <neural_net::set_train_stop_function>, <neural_net::get_train_stop_function>
    */
    enum stop_function_enum {
        STOPFUNC_MSE = FANN_STOPFUNC_MSE,
        STOPFUNC_BIT
    };

    /* Enum: training_algorithm_enum
	    The Training algorithms used when training on <training_data> with functions like
	    <neural_net::train_on_data> or <neural_net::train_on_file>. The incremental training
        looks alters the weights after each time it is presented an input pattern, while batch
        only alters the weights once after it has been presented to all the patterns.

	    TRAIN_INCREMENTAL -  Standard backpropagation algorithm, where the weights are
		    updated after each training pattern. This means that the weights are updated many
		    times during a single epoch. For this reason some problems, will train very fast with
		    this algorithm, while other more advanced problems will not train very well.
	    TRAIN_BATCH -  Standard backpropagation algorithm, where the weights are updated after
		    calculating the mean square error for the whole training set. This means that the weights
		    are only updated once during an epoch. For this reason some problems, will train slower with
		    this algorithm. But since the mean square error is calculated more correctly than in
		    incremental training, some problems will reach a better solutions with this algorithm.
	    TRAIN_RPROP - A more advanced batch training algorithm which achieves good results
		    for many problems. The RPROP training algorithm is adaptive, and does therefore not
		    use the learning_rate. Some other parameters can however be set to change the way the
		    RPROP algorithm works, but it is only recommended for users with insight in how the RPROP
		    training algorithm works. The RPROP training algorithm is described by
		    [Riedmiller and Braun, 1993], but the actual learning algorithm used here is the
		    iRPROP- training algorithm which is described by [Igel and Husken, 2000] which
		    is a variant of the standard RPROP training algorithm.
	    TRAIN_QUICKPROP - A more advanced batch training algorithm which achieves good results
		    for many problems. The quickprop training algorithm uses the learning_rate parameter
		    along with other more advanced parameters, but it is only recommended to change these
		    advanced parameters, for users with insight in how the quickprop training algorithm works.
		    The quickprop training algorithm is described by [Fahlman, 1988].
		FANN_TRAIN_SARPROP - THE SARPROP ALGORITHM: A SIMULATED ANNEALING ENHANCEMENT TO RESILIENT BACK PROPAGATION
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8197&rep=rep1&type=pdf


	    See also:
		    <neural_net::set_training_algorithm>, <neural_net::get_training_algorithm>
    */
    enum training_algorithm_enum {
        TRAIN_INCREMENTAL = FANN_TRAIN_INCREMENTAL,
        TRAIN_BATCH,
        TRAIN_RPROP,
        TRAIN_QUICKPROP,
        TRAIN_SARPROP
    };

    /* Enum: activation_function_enum

	    The activation functions used for the neurons during training. The activation functions
	    can either be defined for a group of neurons by <neural_net::set_activation_function_hidden>
        and <neural_net::set_activation_function_output> or it can be defined for a single neuron by
        <neural_net::set_activation_function>.

	    The steepness of an activation function is defined in the same way by
	    <neural_net::set_activation_steepness_hidden>, <neural_net::set_activation_steepness_output>
        and <neural_net::set_activation_steepness>.

       The functions are described with functions where:
       * x is the input to the activation function,
       * y is the output,
       * s is the steepness and
       * d is the derivation.

       FANN_LINEAR - Linear activation function.
         * span: -inf < y < inf
	     * y = x*s, d = 1*s
	     * Can NOT be used in fixed point.

       FANN_THRESHOLD - Threshold activation function.
	     * x < 0 -> y = 0, x >= 0 -> y = 1
	     * Can NOT be used during training.

       FANN_THRESHOLD_SYMMETRIC - Threshold activation function.
	     * x < 0 -> y = 0, x >= 0 -> y = 1
	     * Can NOT be used during training.

       FANN_SIGMOID - Sigmoid activation function.
	     * One of the most used activation functions.
	     * span: 0 < y < 1
	     * y = 1/(1 + exp(-2*s*x))
	     * d = 2*s*y*(1 - y)

       FANN_SIGMOID_STEPWISE - Stepwise linear approximation to sigmoid.
	     * Faster than sigmoid but a bit less precise.

       FANN_SIGMOID_SYMMETRIC - Symmetric sigmoid activation function, aka. tanh.
	     * One of the most used activation functions.
	     * span: -1 < y < 1
	     * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
	     * d = s*(1-(y*y))

       FANN_SIGMOID_SYMMETRIC - Stepwise linear approximation to symmetric sigmoid.
	     * Faster than symmetric sigmoid but a bit less precise.

       FANN_GAUSSIAN - Gaussian activation function.
	     * 0 when x = -inf, 1 when x = 0 and 0 when x = inf
	     * span: 0 < y < 1
	     * y = exp(-x*s*x*s)
	     * d = -2*x*s*y*s

       FANN_GAUSSIAN_SYMMETRIC - Symmetric gaussian activation function.
	     * -1 when x = -inf, 1 when x = 0 and 0 when x = inf
	     * span: -1 < y < 1
	     * y = exp(-x*s*x*s)*2-1
	     * d = -2*x*s*(y+1)*s

       FANN_ELLIOT - Fast (sigmoid like) activation function defined by David Elliott
	     * span: 0 < y < 1
	     * y = ((x*s) / 2) / (1 + |x*s|) + 0.5
	     * d = s*1/(2*(1+|x*s|)*(1+|x*s|))

       FANN_ELLIOT_SYMMETRIC - Fast (symmetric sigmoid like) activation function defined by David Elliott
	     * span: -1 < y < 1
	     * y = (x*s) / (1 + |x*s|)
	     * d = s*1/((1+|x*s|)*(1+|x*s|))

	    FANN_LINEAR_PIECE - Bounded linear activation function.
	     * span: 0 < y < 1
	     * y = x*s, d = 1*s

	    FANN_LINEAR_PIECE_SYMMETRIC - Bounded Linear activation function.
	     * span: -1 < y < 1
	     * y = x*s, d = 1*s

        FANN_SIN_SYMMETRIC - Periodical sinus activation function.
         * span: -1 <= y <= 1
         * y = sin(x*s)
         * d = s*cos(x*s)

        FANN_COS_SYMMETRIC - Periodical cosinus activation function.
         * span: -1 <= y <= 1
         * y = cos(x*s)
         * d = s*-sin(x*s)

	    See also:
		    <neural_net::set_activation_function_hidden>,
		    <neural_net::set_activation_function_output>
    */
    enum activation_function_enum {
        LINEAR = FANN_LINEAR,
        THRESHOLD,
        THRESHOLD_SYMMETRIC,
        SIGMOID,
        SIGMOID_STEPWISE,
        SIGMOID_SYMMETRIC,
        SIGMOID_SYMMETRIC_STEPWISE,
        GAUSSIAN,
        GAUSSIAN_SYMMETRIC,
        GAUSSIAN_STEPWISE,
        ELLIOT,
        ELLIOT_SYMMETRIC,
        LINEAR_PIECE,
        LINEAR_PIECE_SYMMETRIC,
        SIN_SYMMETRIC,
        COS_SYMMETRIC
    };

    /* Enum: network_type_enum

        Definition of network types used by <neural_net::get_network_type>

        LAYER - Each layer only has connections to the next layer
        SHORTCUT - Each layer has connections to all following layers

       See Also:
          <neural_net::get_network_type>, <fann_get_network_type>

       This enumeration appears in FANN >= 2.1.0
    */
    enum network_type_enum {
        LAYER = FANN_NETTYPE_LAYER,
        SHORTCUT
    };

    /* Type: connection

        Describes a connection between two neurons and its weight

        from_neuron - Unique number used to identify source neuron
        to_neuron - Unique number used to identify destination neuron
        weight - The numerical value of the weight

        See Also:
            <neural_net::get_connection_array>, <neural_net::set_weight_array>

       This structure appears in FANN >= 2.1.0
    */
    typedef struct fann_connection connection;

    /* Forward declaration of class neural_net and training_data */
    class neural_net;

    class training_data;

    /* Type: callback_type
       This callback function can be called during training when using <neural_net::train_on_data>,
       <neural_net::train_on_file> or <neural_net::cascadetrain_on_data>.

        >typedef int (*callback_type) (neural_net &net, training_data &train,
        >    unsigned int max_epochs, unsigned int epochs_between_reports,
        >    float desired_error, unsigned int epochs, void *user_data);

	    The callback can be set by using <neural_net::set_callback> and is very useful for doing custom
	    things during training. It is recommended to use this function when implementing custom
	    training procedures, or when visualizing the training in a GUI etc. The parameters which the
	    callback function takes is the parameters given to the <neural_net::train_on_data>, plus an epochs
	    parameter which tells how many epochs the training have taken so far.

	    The callback function should return an integer, if the callback function returns -1, the training
	    will terminate.

	    Example of a callback function that prints information to cout:
            >int print_callback(FANN::neural_net &net, FANN::training_data &train,
            >    unsigned int max_epochs, unsigned int epochs_between_reports,
            >    float desired_error, unsigned int epochs, void *user_data)
            >{
            >    cout << "Epochs     " << setw(8) << epochs << ". "
            >         << "Current Error: " << left << net.get_MSE() << right << endl;
            >    return 0;
            >}

	    See also:
		    <neural_net::set_callback>, <fann_callback_type>
     */
    typedef int (*callback_type)(neural_net &net, training_data &train,
                                 unsigned int max_epochs, unsigned int epochs_between_reports,
                                 float desired_error, unsigned int epochs, void *user_data);
}

#endif //FANN_FANN_DATA_CPP_H_H
