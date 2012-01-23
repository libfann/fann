#ifndef FANN_CPP_H_INCLUDED
#define FANN_CPP_H_INCLUDED

/*
 *
 *  Fast Artificial Neural Network (fann) C++ Wrapper
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

/*
 *  Title: FANN Wrapper for C++
 *
 *  Overview:
 *
 *  The Fann Wrapper for C++ provides two classes: <neural_net>
 *  and <training_data>. To use the wrapper include
 *  doublefann.h, floatfann.h or fixedfann.h before the
 *  fann_cpp.h header file. To get started see xor_sample.cpp
 *  in the examples directory. The license is LGPL. Copyright (C)
 *  2004-2006 created by <freegoldbar@yahoo.com>.
 *
 *  Note:  Notes and differences from C API
 *
 *  -  The Fann Wrapper for C++ is a minimal wrapper without use of
 *       templates or exception handling for efficient use in any environment.
 *       Benefits include stricter type checking, simpler memory
 *       management and possibly code completion in program editor.
 *  -  Method names are the same as the function names in the C
 *       API except the fann_ prefix has been removed. Enums in the
 *       namespace are similarly defined without the FANN_ prefix.
 *  -  The arguments to the methods are the same as the C API
 *       except that the struct fann *ann/struct fann_train_data *data
 *       arguments are encapsulated so they are not present in the
 *       method signatures or are translated into class references.
 *  -  The various create methods return a boolean set to true to
 *       indicate that the neural network was created, false otherwise.
 *       The same goes for the read_train_from_file method.
 *  -  The neural network and training data is automatically cleaned
 *       up in the destructors and create/read methods.
 *  -  To make the destructors virtual define USE_VIRTUAL_DESTRUCTOR
 *       before including the header file.
 *  -  Additional methods are available on the training_data class to
 *       give access to the underlying training data. They are get_input,
 *       get_output and set_train_data. Finally fann_duplicate_train_data
 *       has been replaced by a copy constructor.
 *
 *  Note: Changes
 *
 *  Version 2.2.0:
 *     - General update to fann C library 2.2.0 with support for new functionality
 *
 *  Version 2.1.0:
 *     - General update to fann C library 2.1.0 with support for new functionality
 *     - Due to changes in the C API the C++ API is not fully backward compatible:
 *        The create methods have changed names and parameters.
 *        The training callback function has different parameters and a set_callback.
 *        Some <training_data> methods have updated names.
 *        Get activation function and steepness is available for neurons, not layers.
 *     - Extensions are now part of fann so there is no fann_extensions.h
 *
 *  Version 1.2.0:
 *     - Changed char pointers to const std::string references
 *     - Added const_casts where the C API required it
 *     - Initialized enums from the C enums instead of numeric constants
 *     - Added a method set_train_data that copies and allocates training
 *     - data in a way that is compatible with the way the C API deallocates
 *     - the data thus making it possible to change training data.
 *     - The get_rprop_increase_factor method did not return its value
 *
 *  Version 1.0.0:
 *     - Initial version
 *
 */

#include <stdarg.h>
#include <string>

/* Namespace: FANN
    The FANN namespace groups the C++ wrapper definitions */
namespace FANN
{
    /* Enum: error_function_enum
	    Error function used during training.
    	
	    ERRORFUNC_LINEAR - Standard linear error function.
	    ERRORFUNC_TANH - Tanh error function, usually better 
		    but can require a lower learning rate. This error function agressively targets outputs that
		    differ much from the desired, while not targetting outputs that only differ a little that much.
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
    enum stop_function_enum
    {
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
		    are only updated once during a epoch. For this reason some problems, will train slower with 
		    this algorithm. But since the mean square error is calculated more correctly than in 
		    incremental training, some problems will reach a better solutions with this algorithm.
	    TRAIN_RPROP - A more advanced batch training algorithm which achieves good results 
		    for many problems. The RPROP training algorithm is adaptive, and does therefore not 
		    use the learning_rate. Some other parameters can however be set to change the way the 
		    RPROP algorithm works, but it is only recommended for users with insight in how the RPROP 
		    training algorithm works. The RPROP training algorithm is described by 
		    [Riedmiller and Braun, 1993], but the actual learning algorithm used here is the 
		    iRPROP- training algorithm which is described by [Igel and Husken, 2000] which 
    	    is an variety of the standard RPROP training algorithm.
	    TRAIN_QUICKPROP - A more advanced batch training algorithm which achieves good results 
		    for many problems. The quickprop training algorithm uses the learning_rate parameter 
		    along with other more advanced parameters, but it is only recommended to change these 
		    advanced parameters, for users with insight in how the quickprop training algorithm works.
		    The quickprop training algorithm is described by [Fahlman, 1988].
    	
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
    enum network_type_enum
    {
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
    	
	    The callback can be set by using <neural_net::set_callback> and is very usefull for doing custom 
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
    typedef int (*callback_type) (neural_net &net, training_data &train,
        unsigned int max_epochs, unsigned int epochs_between_reports,
        float desired_error, unsigned int epochs, void *user_data);

    /*************************************************************************/

    /* Class: training_data

        Encapsulation of a training data set <struct fann_train_data> and
        associated C API functions.
    */
    class training_data
    {
    public:
        /* Constructor: training_data
        
            Default constructor creates an empty neural net.
            Use <read_train_from_file>, <set_train_data> or <create_train_from_callback> to initialize.
        */
        training_data() : train_data(NULL)
        {
        }

        /* Constructor: training_data
        
            Copy constructor constructs a copy of the training data.
            Corresponds to the C API <fann_duplicate_train_data> function.
        */
        training_data(const training_data &data)
        {
            destroy_train();
            if (data.train_data != NULL)
            {
                train_data = fann_duplicate_train_data(data.train_data);
            }
        }

        /* Destructor: ~training_data

            Provides automatic cleanup of data.
            Define USE_VIRTUAL_DESTRUCTOR if you need the destructor to be virtual.

            See also:
                <destroy>
        */
#ifdef USE_VIRTUAL_DESTRUCTOR
        virtual
#endif
        ~training_data()
        {
            destroy_train();
        }

        /* Method: destroy
        
            Destructs the training data. Called automatically by the destructor.

            See also:
                <~training_data>
        */
        void destroy_train()
        {
            if (train_data != NULL)
            {
                fann_destroy_train(train_data);
                train_data = NULL;
            }
        }

        /* Method: read_train_from_file
           Reads a file that stores training data.
           
           The file must be formatted like:
           >num_train_data num_input num_output
           >inputdata seperated by space
           >outputdata seperated by space
           >
           >.
           >.
           >.
           >
           >inputdata seperated by space
           >outputdata seperated by space
           
           See also:
   	        <neural_net::train_on_data>, <save_train>, <fann_read_train_from_file>

            This function appears in FANN >= 1.0.0
        */ 
        bool read_train_from_file(const std::string &filename)
        {
            destroy_train();
            train_data = fann_read_train_from_file(filename.c_str());
            return (train_data != NULL);
        }

        /* Method: save_train
           
           Save the training structure to a file, with the format as specified in <read_train_from_file>

           Return:
           The function returns true on success and false on failure.
              
           See also:
   	        <read_train_from_file>, <save_train_to_fixed>, <fann_save_train>
        	
           This function appears in FANN >= 1.0.0.   	
         */ 
        bool save_train(const std::string &filename)
        {
            if (train_data == NULL)
            {
                return false;
            }
            if (fann_save_train(train_data, filename.c_str()) == -1)
            {
                return false;
            }
            return true;
        }

        /* Method: save_train_to_fixed
           
           Saves the training structure to a fixed point data file.
         
           This function is very usefull for testing the quality of a fixed point network.
           
           Return:
           The function returns true on success and false on failure.
           
           See also:
   	        <save_train>, <fann_save_train_to_fixed>

           This function appears in FANN >= 1.0.0.   	
         */ 
        bool save_train_to_fixed(const std::string &filename, unsigned int decimal_point)
        {
            if (train_data == NULL)
            {
                return false;
            }
            if (fann_save_train_to_fixed(train_data, filename.c_str(), decimal_point) == -1)
            {
                return false;
            }
            return true;
        }

        /* Method: shuffle_train_data
           
           Shuffles training data, randomizing the order. 
           This is recommended for incremental training, while it have no influence during batch training.
           
           This function appears in FANN >= 1.1.0.
         */ 
        void shuffle_train_data()
        {
            if (train_data != NULL)
            {
                fann_shuffle_train_data(train_data);
            }
        }

        /* Method: merge_train_data
           
           Merges the data into the data contained in the <training_data>.
           
           This function appears in FANN >= 1.1.0.
         */ 
        void merge_train_data(const training_data &data)
        {
            fann_train_data *new_data = fann_merge_train_data(train_data, data.train_data);
            if (new_data != NULL)
            {
                destroy_train();
                train_data = new_data;
            }
        }

        /* Method: length_train_data
           
           Returns the number of training patterns in the <training_data>.

           See also:
           <num_input_train_data>, <num_output_train_data>, <fann_length_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        unsigned int length_train_data()
        {
            if (train_data == NULL)
            {
                return 0;
            }
            else
            {
                return fann_length_train_data(train_data);
            }
        }

        /* Method: num_input_train_data

           Returns the number of inputs in each of the training patterns in the <training_data>.
           
           See also:
           <num_output_train_data>, <length_train_data>, <fann_num_input_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        unsigned int num_input_train_data()
        {
            if (train_data == NULL)
            {
                return 0;
            }
            else
            {
                return fann_num_input_train_data(train_data);
            }
        }

        /* Method: num_output_train_data
           
           Returns the number of outputs in each of the training patterns in the <struct fann_train_data>.
           
           See also:
           <num_input_train_data>, <length_train_data>, <fann_num_output_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        unsigned int num_output_train_data()
        {
            if (train_data == NULL)
            {
                return 0;
            }
            else
            {
                return fann_num_output_train_data(train_data);
            }
        }

        /* Grant access to the encapsulated data since many situations
            and applications creates the data from sources other than files
            or uses the training data for testing and related functions */

        /* Method: get_input
        
            Returns:
                A pointer to the array of input training data

            See also:
                <get_output>, <set_train_data>
        */
        fann_type **get_input()
        {
            if (train_data == NULL)
            {
                return NULL;
            }
            else
            {
                return train_data->input;
            }
        }

        /* Method: get_output
        
            Returns:
                A pointer to the array of output training data

            See also:
                <get_input>, <set_train_data>
        */
        fann_type **get_output()
        {
            if (train_data == NULL)
            {
                return NULL;
            }
            else
            {
                return train_data->output;
            }
        }

        /* Method: set_train_data

            Set the training data to the input and output data provided.

            A copy of the data is made so there are no restrictions on the
            allocation of the input/output data and the caller is responsible
            for the deallocation of the data pointed to by input and output.

           Parameters:
             num_data      - The number of training data
             num_input     - The number of inputs per training data
             num_output    - The number of ouputs per training data
             input      - The set of inputs (a pointer to an array of pointers to arrays of floating point data)
             output     - The set of desired outputs (a pointer to an array of pointers to arrays of floating point data)

            See also:
                <get_input>, <get_output>
        */
        void set_train_data(unsigned int num_data,
            unsigned int num_input, fann_type **input,
            unsigned int num_output, fann_type **output)
        {
            // Uses the allocation method used in fann
            struct fann_train_data *data =
                (struct fann_train_data *)malloc(sizeof(struct fann_train_data));
            data->input = (fann_type **)calloc(num_data, sizeof(fann_type *));
            data->output = (fann_type **)calloc(num_data, sizeof(fann_type *));

            data->num_data = num_data;
            data->num_input = num_input;
            data->num_output = num_output;

        	fann_type *data_input = (fann_type *)calloc(num_input*num_data, sizeof(fann_type));
        	fann_type *data_output = (fann_type *)calloc(num_output*num_data, sizeof(fann_type));

            for (unsigned int i = 0; i < num_data; ++i)
            {
                data->input[i] = data_input;
                data_input += num_input;
                for (unsigned int j = 0; j < num_input; ++j)
                {
                    data->input[i][j] = input[i][j];
                }
                data->output[i] = data_output;
		        data_output += num_output;
                for (unsigned int j = 0; j < num_output; ++j)
                {
                    data->output[i][j] = output[i][j];
                }
            }
            set_train_data(data);
        }

private:
        /* Set the training data to the struct fann_training_data pointer.
            The struct has to be allocated with malloc to be compatible
            with fann_destroy. */
        void set_train_data(struct fann_train_data *data)
        {
            destroy_train();
            train_data = data;
        }

public:
        /*********************************************************************/

        /* Method: create_train_from_callback
           Creates the training data struct from a user supplied function.
           As the training data are numerable (data 1, data 2...), the user must write
           a function that receives the number of the training data set (input,output)
           and returns the set.

           Parameters:
             num_data      - The number of training data
             num_input     - The number of inputs per training data
             num_output    - The number of ouputs per training data
             user_function - The user suplied function

           Parameters for the user function:
             num        - The number of the training data set
             num_input  - The number of inputs per training data
             num_output - The number of ouputs per training data
             input      - The set of inputs
             output     - The set of desired outputs
          
           See also:
             <training_data::read_train_from_file>, <neural_net::train_on_data>,
             <fann_create_train_from_callback>

            This function appears in FANN >= 2.1.0
        */ 
        void create_train_from_callback(unsigned int num_data,
                                                  unsigned int num_input,
                                                  unsigned int num_output,
                                                  void (FANN_API *user_function)( unsigned int,
                                                                         unsigned int,
                                                                         unsigned int,
                                                                         fann_type * ,
                                                                         fann_type * ))
        {
            destroy_train();
            train_data = fann_create_train_from_callback(num_data, num_input, num_output, user_function);
        }

        /* Method: scale_input_train_data
           
           Scales the inputs in the training data to the specified range.

           See also:
   	        <scale_output_train_data>, <scale_train_data>, <fann_scale_input_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        void scale_input_train_data(fann_type new_min, fann_type new_max)
        {
            if (train_data != NULL)
            {
                fann_scale_input_train_data(train_data, new_min, new_max);
            }
        }

        /* Method: scale_output_train_data
           
           Scales the outputs in the training data to the specified range.

           See also:
   	        <scale_input_train_data>, <scale_train_data>, <fann_scale_output_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        void scale_output_train_data(fann_type new_min, fann_type new_max)
        {
            if (train_data != NULL)
            {
                fann_scale_output_train_data(train_data, new_min, new_max);
            }
        }

        /* Method: scale_train_data
           
           Scales the inputs and outputs in the training data to the specified range.
           
           See also:
   	        <scale_output_train_data>, <scale_input_train_data>, <fann_scale_train_data>

           This function appears in FANN >= 2.0.0.
         */ 
        void scale_train_data(fann_type new_min, fann_type new_max)
        {
            if (train_data != NULL)
            {
                fann_scale_train_data(train_data, new_min, new_max);
            }
        }

        /* Method: subset_train_data
           
           Changes the training data to a subset, starting at position *pos* 
           and *length* elements forward. Use the copy constructor to work
           on a new copy of the training data.
           
            >FANN::training_data full_data_set;
            >full_data_set.read_train_from_file("somefile.train");
            >FANN::training_data *small_data_set = new FANN::training_data(full_data_set);
            >small_data_set->subset_train_data(0, 2); // Only use first two
            >// Use small_data_set ...
            >delete small_data_set;

           See also:
   	        <fann_subset_train_data>

           This function appears in FANN >= 2.0.0.
         */
        void subset_train_data(unsigned int pos, unsigned int length)
        {
            if (train_data != NULL)
            {
                struct fann_train_data *temp = fann_subset_train_data(train_data, pos, length);
                destroy_train();
                train_data = temp;
            }
        }

        /*********************************************************************/

    protected:
        /* The neural_net class has direct access to the training data */
        friend class neural_net;

        /* Pointer to the encapsulated training data */
        struct fann_train_data* train_data;
    };

    /*************************************************************************/

    /* Class: neural_net

        Encapsulation of a neural network <struct fann> and
        associated C API functions.
    */
    class neural_net
    {
    public:
        /* Constructor: neural_net
        
            Default constructor creates an empty neural net.
            Use one of the create functions to create the neural network.

            See also:
		        <create_standard>, <create_sparse>, <create_shortcut>,
		        <create_standard_array>, <create_sparse_array>, <create_shortcut_array>
        */
        neural_net() : ann(NULL)
        {
        }

	/* Constructor neural_net

	    Creates a copy the other neural_net.
            
	    See also:
	    		<copy_from_struct_fann>
        */
	neural_net(const neural_net& other)
	{
	    copy_from_struct_fann(other.ann);
	}
	
	/* Constructor: neural_net

	   Creates a copy the other neural_net.
	    
	   See also:
	    		<copy_from_struct_fann>
        */
	neural_net(struct fann* other)
	{
	    copy_from_struct_fann(other);
	}

	/* Method: copy_from_struct_fann
	   
	   Set the internal fann struct to a copy of other
	*/
	void copy_from_struct_fann(struct fann* other)
	{
	    destroy();
	    if (other != NULL)
		ann=fann_copy(other);
	}

        /* Destructor: ~neural_net

            Provides automatic cleanup of data.
            Define USE_VIRTUAL_DESTRUCTOR if you need the destructor to be virtual.

            See also:
                <destroy>
        */
#ifdef USE_VIRTUAL_DESTRUCTOR
        virtual
#endif
        ~neural_net()
        {
            destroy();
        }

        /* Method: destroy
        
            Destructs the entire network. Called automatically by the destructor.

            See also:
                <~neural_net>
        */
        void destroy()
        {
            if (ann != NULL)
            {
                user_context *user_data = static_cast<user_context *>(fann_get_user_data(ann));
                if (user_data != NULL)
                    delete user_data;

                fann_destroy(ann);
                ann = NULL;
            }
        }

        /* Method: create_standard
        	
	        Creates a standard fully connected backpropagation neural network.

	        There will be a bias neuron in each layer (except the output layer),
	        and this bias neuron will be connected to all neurons in the next layer.
	        When running the network, the bias nodes always emits 1.
        	
	        Parameters:
		        num_layers - The total number of layers including the input and the output layer.
		        ... - Integer values determining the number of neurons in each layer starting with the 
			        input layer and ending with the output layer.
        			
	        Returns:
		        Boolean true if the network was created, false otherwise.

            Example:
                >const unsigned int num_layers = 3;
                >const unsigned int num_input = 2;
                >const unsigned int num_hidden = 3;
                >const unsigned int num_output = 1;
                >
                >FANN::neural_net net;
                >net.create_standard(num_layers, num_input, num_hidden, num_output);

	        See also:
		        <create_standard_array>, <create_sparse>, <create_shortcut>,
		        <fann_create_standard_array>

	        This function appears in FANN >= 2.0.0.
        */ 
        bool create_standard(unsigned int num_layers, ...)
        {
            va_list layers;
            va_start(layers, num_layers);
            bool status = create_standard_array(num_layers,
                reinterpret_cast<const unsigned int *>(layers));
            va_end(layers);
            return status;
        }

        /* Method: create_standard_array

           Just like <create_standard>, but with an array of layer sizes
           instead of individual parameters.

	        See also:
		        <create_standard>, <create_sparse>, <create_shortcut>,
		        <fann_create_standard>

	        This function appears in FANN >= 2.0.0.
        */ 
        bool create_standard_array(unsigned int num_layers, const unsigned int * layers)
        {
            destroy();
            ann = fann_create_standard_array(num_layers, layers);
            return (ann != NULL);
        }

        /* Method: create_sparse

	        Creates a standard backpropagation neural network, which is not fully connected.

	        Parameters:
		        connection_rate - The connection rate controls how many connections there will be in the
   			        network. If the connection rate is set to 1, the network will be fully
   			        connected, but if it is set to 0.5 only half of the connections will be set.
			        A connection rate of 1 will yield the same result as <fann_create_standard>
		        num_layers - The total number of layers including the input and the output layer.
		        ... - Integer values determining the number of neurons in each layer starting with the 
			        input layer and ending with the output layer.
        			
	        Returns:
		        Boolean true if the network was created, false otherwise.

	        See also:
		        <create_standard>, <create_sparse_array>, <create_shortcut>,
		        <fann_create_sparse>

	        This function appears in FANN >= 2.0.0.
        */
        bool create_sparse(float connection_rate, unsigned int num_layers, ...)
        {
            va_list layers;
            va_start(layers, num_layers);
            bool status = create_sparse_array(connection_rate, num_layers,
                reinterpret_cast<const unsigned int *>(layers));
            va_end(layers);
            return status;
        }

        /* Method: create_sparse_array
           Just like <create_sparse>, but with an array of layer sizes
           instead of individual parameters.

           See <create_sparse> for a description of the parameters.

	        See also:
		        <create_standard>, <create_sparse>, <create_shortcut>,
		        <fann_create_sparse_array>

	        This function appears in FANN >= 2.0.0.
        */
        bool create_sparse_array(float connection_rate,
            unsigned int num_layers, const unsigned int * layers)
        {
            destroy();
            ann = fann_create_sparse_array(connection_rate, num_layers, layers);
            return (ann != NULL);
        }

        /* Method: create_shortcut

	        Creates a standard backpropagation neural network, which is not fully connected and which
	        also has shortcut connections.

 	        Shortcut connections are connections that skip layers. A fully connected network with shortcut 
	        connections, is a network where all neurons are connected to all neurons in later layers. 
	        Including direct connections from the input layer to the output layer.

	        See <create_standard> for a description of the parameters.

	        See also:
		        <create_standard>, <create_sparse>, <create_shortcut_array>,
		        <fann_create_shortcut>

	        This function appears in FANN >= 2.0.0.
        */ 
        bool create_shortcut(unsigned int num_layers, ...)
        {
            va_list layers;
            va_start(layers, num_layers);
            bool status = create_shortcut_array(num_layers,
                reinterpret_cast<const unsigned int *>(layers));
            va_end(layers);
            return status;
        }

        /* Method: create_shortcut_array

           Just like <create_shortcut>, but with an array of layer sizes
           instead of individual parameters.

	        See <create_standard_array> for a description of the parameters.

	        See also:
		        <create_standard>, <create_sparse>, <create_shortcut>,
		        <fann_create_shortcut_array>

	        This function appears in FANN >= 2.0.0.
        */
        bool create_shortcut_array(unsigned int num_layers,
            const unsigned int * layers)
        {
            destroy();
            ann = fann_create_shortcut_array(num_layers, layers);
            return (ann != NULL);
        }

        /* Method: run

	        Will run input through the neural network, returning an array of outputs, the number of which being 
	        equal to the number of neurons in the output layer.

	        See also:
		        <test>, <fann_run>

	        This function appears in FANN >= 1.0.0.
        */ 
        fann_type* run(fann_type *input)
        {
            if (ann == NULL)
            {
                return NULL;
            }
            return fann_run(ann, input);
        }

        /* Method: randomize_weights

	        Give each connection a random weight between *min_weight* and *max_weight*
           
	        From the beginning the weights are random between -0.1 and 0.1.

	        See also:
		        <init_weights>, <fann_randomize_weights>

	        This function appears in FANN >= 1.0.0.
        */ 
        void randomize_weights(fann_type min_weight, fann_type max_weight)
        {
            if (ann != NULL)
            {
                fann_randomize_weights(ann, min_weight, max_weight);
            }
        }

        /* Method: init_weights

  	        Initialize the weights using Widrow + Nguyen's algorithm.
        	
 	        This function behaves similarly to fann_randomize_weights. It will use the algorithm developed 
	        by Derrick Nguyen and Bernard Widrow to set the weights in such a way 
	        as to speed up training. This technique is not always successful, and in some cases can be less 
	        efficient than a purely random initialization.

	        The algorithm requires access to the range of the input data (ie, largest and smallest input), 
	        and therefore accepts a second argument, data, which is the training data that will be used to 
	        train the network.

	        See also:
		        <randomize_weights>, <training_data::read_train_from_file>,
                <fann_init_weights>

	        This function appears in FANN >= 1.1.0.
        */ 
        void init_weights(const training_data &data)
        {
            if ((ann != NULL) && (data.train_data != NULL))
            {
                fann_init_weights(ann, data.train_data);
            }
        }

        /* Method: print_connections

	        Will print the connections of the ann in a compact matrix, for easy viewing of the internals 
	        of the ann.

	        The output from fann_print_connections on a small (2 2 1) network trained on the xor problem
	        >Layer / Neuron 012345
	        >L   1 / N    3 BBa...
	        >L   1 / N    4 BBA...
	        >L   1 / N    5 ......
	        >L   2 / N    6 ...BBA
	        >L   2 / N    7 ......
        		  
	        This network have five real neurons and two bias neurons. This gives a total of seven neurons 
	        named from 0 to 6. The connections between these neurons can be seen in the matrix. "." is a 
	        place where there is no connection, while a character tells how strong the connection is on a 
	        scale from a-z. The two real neurons in the hidden layer (neuron 3 and 4 in layer 1) has 
	        connection from the three neurons in the previous layer as is visible in the first two lines. 
	        The output neuron (6) has connections form the three neurons in the hidden layer 3 - 5 as is 
	        visible in the fourth line.

	        To simplify the matrix output neurons is not visible as neurons that connections can come from, 
	        and input and bias neurons are not visible as neurons that connections can go to.

	        This function appears in FANN >= 1.2.0.
        */ 
        void print_connections()
        {
            if (ann != NULL)
            {
                fann_print_connections(ann);
            }
        }

        /* Method: create_from_file
           
           Constructs a backpropagation neural network from a configuration file,
           which have been saved by <save>.
           
           See also:
   	        <save>, <save_to_fixed>, <fann_create_from_file>
           	
           This function appears in FANN >= 1.0.0.
         */
        bool create_from_file(const std::string &configuration_file)
        {
            destroy();
            ann = fann_create_from_file(configuration_file.c_str());
            return (ann != NULL);
        }

        /* Method: save

           Save the entire network to a configuration file.
           
           The configuration file contains all information about the neural network and enables 
           <create_from_file> to create an exact copy of the neural network and all of the
           parameters associated with the neural network.
           
           These two parameters (<set_callback>, <set_error_log>) are *NOT* saved 
           to the file because they cannot safely be ported to a different location. Also temporary
           parameters generated during training like <get_MSE> is not saved.
           
           Return:
           The function returns 0 on success and -1 on failure.
           
           See also:
            <create_from_file>, <save_to_fixed>, <fann_save>

           This function appears in FANN >= 1.0.0.
         */
        bool save(const std::string &configuration_file)
        {
            if (ann == NULL)
            {
                return false;
            }
            if (fann_save(ann, configuration_file.c_str()) == -1)
            {
                return false;
            }
            return true;
        }

        /* Method: save_to_fixed

           Saves the entire network to a configuration file.
           But it is saved in fixed point format no matter which
           format it is currently in.

           This is usefull for training a network in floating points,
           and then later executing it in fixed point.

           The function returns the bit position of the fix point, which
           can be used to find out how accurate the fixed point network will be.
           A high value indicates high precision, and a low value indicates low
           precision.

           A negative value indicates very low precision, and a very
           strong possibility for overflow.
           (the actual fix point will be set to 0, since a negative
           fix point does not make sence).

           Generally, a fix point lower than 6 is bad, and should be avoided.
           The best way to avoid this, is to have less connections to each neuron,
           or just less neurons in each layer.

           The fixed point use of this network is only intended for use on machines that
           have no floating point processor, like an iPAQ. On normal computers the floating
           point version is actually faster.

           See also:
            <create_from_file>, <save>, <fann_save_to_fixed>

           This function appears in FANN >= 1.0.0.
        */ 
        int save_to_fixed(const std::string &configuration_file)
        {
            int fixpoint = 0;
            if (ann != NULL)
            {
                fixpoint = fann_save_to_fixed(ann, configuration_file.c_str());
            }
            return fixpoint;
        }

#ifndef FIXEDFANN
        /* Method: train

           Train one iteration with a set of inputs, and a set of desired outputs.
           This training is always incremental training (see <FANN::training_algorithm_enum>),
           since only one pattern is presented.
           
           Parameters:
   	        ann - The neural network structure
   	        input - an array of inputs. This array must be exactly <fann_get_num_input> long.
   	        desired_output - an array of desired outputs. This array must be exactly <fann_get_num_output> long.
           	
   	        See also:
   		        <train_on_data>, <train_epoch>, <fann_train>
           	
   	        This function appears in FANN >= 1.0.0.
         */ 
        void train(fann_type *input, fann_type *desired_output)
        {
            if (ann != NULL)
            {
                fann_train(ann, input, desired_output);
            }
        }

        /* Method: train_epoch
            Train one epoch with a set of training data.
           
            Train one epoch with the training data stored in data. One epoch is where all of 
            the training data is considered exactly once.

	        This function returns the MSE error as it is calculated either before or during 
	        the actual training. This is not the actual MSE after the training epoch, but since 
	        calculating this will require to go through the entire training set once more, it is 
	        more than adequate to use this value during training.

	        The training algorithm used by this function is chosen by the <fann_set_training_algorithm> 
	        function.
        	
	        See also:
		        <train_on_data>, <test_data>, <fann_train_epoch>
        		
	        This function appears in FANN >= 1.2.0.
         */ 
        float train_epoch(const training_data &data)
        {
            float mse = 0.0f;
            if ((ann != NULL) && (data.train_data != NULL))
            {
                mse = fann_train_epoch(ann, data.train_data);
            }
            return mse;
        }

        /* Method: train_on_data

           Trains on an entire dataset, for a period of time. 
           
           This training uses the training algorithm chosen by <set_training_algorithm>,
           and the parameters set for these training algorithms.
           
           Parameters:
   		        ann - The neural network
   		        data - The data, which should be used during training
   		        max_epochs - The maximum number of epochs the training should continue
   		        epochs_between_reports - The number of epochs between printing a status report to stdout.
   			        A value of zero means no reports should be printed.
   		        desired_error - The desired <get_MSE> or <get_bit_fail>, depending on which stop function
   			        is chosen by <set_train_stop_function>.

	        Instead of printing out reports every epochs_between_reports, a callback function can be called 
	        (see <set_callback>).
        	
	        See also:
		        <train_on_file>, <train_epoch>, <fann_train_on_data>

	        This function appears in FANN >= 1.0.0.
        */ 
        void train_on_data(const training_data &data, unsigned int max_epochs,
            unsigned int epochs_between_reports, float desired_error)
        {
            if ((ann != NULL) && (data.train_data != NULL))
            {
                fann_train_on_data(ann, data.train_data, max_epochs,
                    epochs_between_reports, desired_error);
            }
        }

        /* Method: train_on_file
           
           Does the same as <train_on_data>, but reads the training data directly from a file.
           
           See also:
   		        <train_on_data>, <fann_train_on_file>

	        This function appears in FANN >= 1.0.0.
        */ 
        void train_on_file(const std::string &filename, unsigned int max_epochs,
            unsigned int epochs_between_reports, float desired_error)
        {
            if (ann != NULL)
            {
                fann_train_on_file(ann, filename.c_str(),
                    max_epochs, epochs_between_reports, desired_error);
            }
        }
#endif /* NOT FIXEDFANN */

        /* Method: test

           Test with a set of inputs, and a set of desired outputs.
           This operation updates the mean square error, but does not
           change the network in any way.
           
           See also:
   		        <test_data>, <train>, <fann_test>
           
           This function appears in FANN >= 1.0.0.
        */ 
        fann_type * test(fann_type *input, fann_type *desired_output)
        {
            fann_type * output = NULL;
            if (ann != NULL)
            {
                output = fann_test(ann, input, desired_output);
            }
            return output;
        }

        /* Method: test_data
          
           Test a set of training data and calculates the MSE for the training data. 
           
           This function updates the MSE and the bit fail values.
           
           See also:
 	        <test>, <get_MSE>, <get_bit_fail>, <fann_test_data>

	        This function appears in FANN >= 1.2.0.
         */ 
        float test_data(const training_data &data)
        {
            float mse = 0.0f;
            if ((ann != NULL) && (data.train_data != NULL))
            {
                mse = fann_test_data(ann, data.train_data);
            }
            return mse;
        }

        /* Method: get_MSE
           Reads the mean square error from the network.
           
           Reads the mean square error from the network. This value is calculated during 
           training or testing, and can therefore sometimes be a bit off if the weights 
           have been changed since the last calculation of the value.
           
           See also:
   	        <test_data>, <fann_get_MSE>

	        This function appears in FANN >= 1.1.0.
         */ 
        float get_MSE()
        {
            float mse = 0.0f;
            if (ann != NULL)
            {
                mse = fann_get_MSE(ann);
            }
            return mse;
        }

        /* Method: reset_MSE

           Resets the mean square error from the network.
   
           This function also resets the number of bits that fail.
           
           See also:
   	        <get_MSE>, <get_bit_fail_limit>, <fann_reset_MSE>
           
            This function appears in FANN >= 1.1.0
         */ 
        void reset_MSE()
        {
            if (ann != NULL)
            {
                fann_reset_MSE(ann);
            }
        }

        /* Method: set_callback
           
           Sets the callback function for use during training. The user_data is passed to
           the callback. It can point to arbitrary data that the callback might require and
           can be NULL if it is not used.
         	
           See <FANN::callback_type> for more information about the callback function.
           
           The default callback function simply prints out some status information.

           This function appears in FANN >= 2.0.0.
         */
        void set_callback(callback_type callback, void *user_data)
        {
            if (ann != NULL)
            {
                // Allocated data is also deleted in the destroy method called by the destructor
                user_context *user_instance = static_cast<user_context *>(fann_get_user_data(ann));
                if (user_instance != NULL)
                    delete user_instance;

                user_instance = new user_context();
                user_instance->user_callback = callback;
                user_instance->user_data = user_data;
                user_instance->net = this;
                fann_set_user_data(ann, user_instance);

                if (callback != NULL)
                    fann_set_callback(ann, &FANN::neural_net::internal_callback);
                else
                    fann_set_callback(ann, NULL);
            }
        }

        /* Method: print_parameters

  	        Prints all of the parameters and options of the neural network

            See also:
                <fann_print_parameters>

	        This function appears in FANN >= 1.2.0.
        */ 
        void print_parameters()
        {
            if (ann != NULL)
            {
                fann_print_parameters(ann);
            }
        }

        /* Method: get_training_algorithm

           Return the training algorithm as described by <FANN::training_algorithm_enum>.
           This training algorithm is used by <train_on_data> and associated functions.
           
           Note that this algorithm is also used during <cascadetrain_on_data>, although only
           FANN::TRAIN_RPROP and FANN::TRAIN_QUICKPROP is allowed during cascade training.
           
           The default training algorithm is FANN::TRAIN_RPROP.
           
           See also:
            <set_training_algorithm>, <FANN::training_algorithm_enum>,
            <fann_get_training_algorithm>

           This function appears in FANN >= 1.0.0.   	
         */ 
        training_algorithm_enum get_training_algorithm()
        {
            fann_train_enum training_algorithm = FANN_TRAIN_INCREMENTAL;
            if (ann != NULL)
            {
                training_algorithm = fann_get_training_algorithm(ann);
            }
            return static_cast<training_algorithm_enum>(training_algorithm);
        }

        /* Method: set_training_algorithm

           Set the training algorithm.
           
           More info available in <get_training_algorithm>

           This function appears in FANN >= 1.0.0.   	
         */ 
        void set_training_algorithm(training_algorithm_enum training_algorithm)
        {
            if (ann != NULL)
            {
                fann_set_training_algorithm(ann,
					static_cast<fann_train_enum>(training_algorithm));
            }
        }

        /* Method: get_learning_rate

           Return the learning rate.
           
           The learning rate is used to determine how aggressive training should be for some of the
           training algorithms (FANN::TRAIN_INCREMENTAL, FANN::TRAIN_BATCH, FANN::TRAIN_QUICKPROP).
           Do however note that it is not used in FANN::TRAIN_RPROP.
           
           The default learning rate is 0.7.
           
           See also:
   	        <set_learning_rate>, <set_training_algorithm>,
            <fann_get_learning_rate>
           
           This function appears in FANN >= 1.0.0.   	
         */ 
        float get_learning_rate()
        {
            float learning_rate = 0.0f;
            if (ann != NULL)
            {
                learning_rate = fann_get_learning_rate(ann);
            }
            return learning_rate;
        }

        /* Method: set_learning_rate

           Set the learning rate.
           
           More info available in <get_learning_rate>

           This function appears in FANN >= 1.0.0.   	
         */ 
        void set_learning_rate(float learning_rate)
        {
            if (ann != NULL)
            {
                fann_set_learning_rate(ann, learning_rate);
            }
        }

        /*************************************************************************************************************/

        /* Method: get_activation_function

           Get the activation function for neuron number *neuron* in layer number *layer*, 
           counting the input layer as layer 0. 
           
           It is not possible to get activation functions for the neurons in the input layer.
           
           Information about the individual activation functions is available at <FANN::activation_function_enum>.

           Returns:
            The activation function for the neuron or -1 if the neuron is not defined in the neural network.
           
           See also:
   	        <set_activation_function_layer>, <set_activation_function_hidden>,
   	        <set_activation_function_output>, <set_activation_steepness>,
            <set_activation_function>, <fann_get_activation_function>

           This function appears in FANN >= 2.1.0
         */ 
        activation_function_enum get_activation_function(int layer, int neuron)
        {
            unsigned int activation_function = 0;
            if (ann != NULL)
            {
                activation_function = fann_get_activation_function(ann, layer, neuron);
            }
            return static_cast<activation_function_enum>(activation_function);
        }

        /* Method: set_activation_function

           Set the activation function for neuron number *neuron* in layer number *layer*, 
           counting the input layer as layer 0. 
           
           It is not possible to set activation functions for the neurons in the input layer.
           
           When choosing an activation function it is important to note that the activation 
           functions have different range. FANN::SIGMOID is e.g. in the 0 - 1 range while 
           FANN::SIGMOID_SYMMETRIC is in the -1 - 1 range and FANN::LINEAR is unbound.
           
           Information about the individual activation functions is available at <FANN::activation_function_enum>.
           
           The default activation function is FANN::SIGMOID_STEPWISE.
           
           See also:
   	        <set_activation_function_layer>, <set_activation_function_hidden>,
   	        <set_activation_function_output>, <set_activation_steepness>,
            <get_activation_function>, <fann_set_activation_function>

           This function appears in FANN >= 2.0.0.
         */ 
        void set_activation_function(activation_function_enum activation_function, int layer, int neuron)
        {
            if (ann != NULL)
            {
                fann_set_activation_function(ann,
					static_cast<fann_activationfunc_enum>(activation_function), layer, neuron);
            }
        }

        /* Method: set_activation_function_layer

           Set the activation function for all the neurons in the layer number *layer*, 
           counting the input layer as layer 0. 
           
           It is not possible to set activation functions for the neurons in the input layer.

           See also:
   	        <set_activation_function>, <set_activation_function_hidden>,
   	        <set_activation_function_output>, <set_activation_steepness_layer>,
            <fann_set_activation_function_layer>

           This function appears in FANN >= 2.0.0.
         */ 
        void set_activation_function_layer(activation_function_enum activation_function, int layer)
        {
            if (ann != NULL)
            {
                fann_set_activation_function_layer(ann,
					static_cast<fann_activationfunc_enum>(activation_function), layer);
            }
        }

        /* Method: set_activation_function_hidden

           Set the activation function for all of the hidden layers.

           See also:
   	        <set_activation_function>, <set_activation_function_layer>,
   	        <set_activation_function_output>, <set_activation_steepness_hidden>,
            <fann_set_activation_function_hidden>

           This function appears in FANN >= 1.0.0.
         */ 
        void set_activation_function_hidden(activation_function_enum activation_function)
        {
            if (ann != NULL)
            {
                fann_set_activation_function_hidden(ann,
					static_cast<fann_activationfunc_enum>(activation_function));
            }
        }

        /* Method: set_activation_function_output

           Set the activation function for the output layer.

           See also:
   	        <set_activation_function>, <set_activation_function_layer>,
   	        <set_activation_function_hidden>, <set_activation_steepness_output>,
            <fann_set_activation_function_output>

           This function appears in FANN >= 1.0.0.
         */ 
        void set_activation_function_output(activation_function_enum activation_function)
        {
            if (ann != NULL)
            {
                fann_set_activation_function_output(ann,
					static_cast<fann_activationfunc_enum>(activation_function));
            }
        }

        /* Method: get_activation_steepness

           Get the activation steepness for neuron number *neuron* in layer number *layer*, 
           counting the input layer as layer 0. 
           
           It is not possible to get activation steepness for the neurons in the input layer.
           
           The steepness of an activation function says something about how fast the activation function 
           goes from the minimum to the maximum. A high value for the activation function will also
           give a more agressive training.
           
           When training neural networks where the output values should be at the extremes (usually 0 and 1, 
           depending on the activation function), a steep activation function can be used (e.g. 1.0).
           
           The default activation steepness is 0.5.
           
           Returns:
            The activation steepness for the neuron or -1 if the neuron is not defined in the neural network.
           
           See also:
   	        <set_activation_steepness_layer>, <set_activation_steepness_hidden>,
   	        <set_activation_steepness_output>, <set_activation_function>,
            <set_activation_steepness>, <fann_get_activation_steepness>

           This function appears in FANN >= 2.1.0
         */ 
        fann_type get_activation_steepness(int layer, int neuron)
        {
            fann_type activation_steepness = 0;
            if (ann != NULL)
            {
                activation_steepness = fann_get_activation_steepness(ann, layer, neuron);
            }
            return activation_steepness;
        }

        /* Method: set_activation_steepness

           Set the activation steepness for neuron number *neuron* in layer number *layer*, 
           counting the input layer as layer 0. 
           
           It is not possible to set activation steepness for the neurons in the input layer.
           
           The steepness of an activation function says something about how fast the activation function 
           goes from the minimum to the maximum. A high value for the activation function will also
           give a more agressive training.
           
           When training neural networks where the output values should be at the extremes (usually 0 and 1, 
           depending on the activation function), a steep activation function can be used (e.g. 1.0).
           
           The default activation steepness is 0.5.
           
           See also:
   	        <set_activation_steepness_layer>, <set_activation_steepness_hidden>,
   	        <set_activation_steepness_output>, <set_activation_function>,
            <get_activation_steepness>, <fann_set_activation_steepness>

           This function appears in FANN >= 2.0.0.
         */ 
        void set_activation_steepness(fann_type steepness, int layer, int neuron)
        {
            if (ann != NULL)
            {
                fann_set_activation_steepness(ann, steepness, layer, neuron);
            }
        }

        /* Method: set_activation_steepness_layer

           Set the activation steepness all of the neurons in layer number *layer*, 
           counting the input layer as layer 0. 
           
           It is not possible to set activation steepness for the neurons in the input layer.
           
           See also:
   	        <set_activation_steepness>, <set_activation_steepness_hidden>,
   	        <set_activation_steepness_output>, <set_activation_function_layer>,
            <fann_set_activation_steepness_layer>

           This function appears in FANN >= 2.0.0.
         */ 
        void set_activation_steepness_layer(fann_type steepness, int layer)
        {
            if (ann != NULL)
            {
                fann_set_activation_steepness_layer(ann, steepness, layer);
            }
        }

        /* Method: set_activation_steepness_hidden

           Set the steepness of the activation steepness in all of the hidden layers.

           See also:
   	        <set_activation_steepness>, <set_activation_steepness_layer>,
   	        <set_activation_steepness_output>, <set_activation_function_hidden>,
            <fann_set_activation_steepness_hidden>

           This function appears in FANN >= 1.2.0.
         */ 
        void set_activation_steepness_hidden(fann_type steepness)
        {
            if (ann != NULL)
            {
                fann_set_activation_steepness_hidden(ann, steepness);
            }
        }

        /* Method: set_activation_steepness_output

           Set the steepness of the activation steepness in the output layer.

           See also:
   	        <set_activation_steepness>, <set_activation_steepness_layer>,
   	        <set_activation_steepness_hidden>, <set_activation_function_output>,
            <fann_set_activation_steepness_output>

           This function appears in FANN >= 1.2.0.
         */ 
        void set_activation_steepness_output(fann_type steepness)
        {
            if (ann != NULL)
            {
                fann_set_activation_steepness_output(ann, steepness);
            }
        }

        /*************************************************************************************************************/

        /* Method: get_train_error_function

           Returns the error function used during training.

           The error functions is described further in <FANN::error_function_enum>
           
           The default error function is FANN::ERRORFUNC_TANH
           
           See also:
   	        <set_train_error_function>, <fann_get_train_error_function>
              
           This function appears in FANN >= 1.2.0.
          */ 
        error_function_enum get_train_error_function()
        {
            fann_errorfunc_enum train_error_function = FANN_ERRORFUNC_LINEAR;
            if (ann != NULL)
            {
                train_error_function = fann_get_train_error_function(ann);
            }
            return static_cast<error_function_enum>(train_error_function);
        }

        /* Method: set_train_error_function

           Set the error function used during training.
           
           The error functions is described further in <FANN::error_function_enum>
           
           See also:
   	        <get_train_error_function>, <fann_set_train_error_function>
              
           This function appears in FANN >= 1.2.0.
         */ 
        void set_train_error_function(error_function_enum train_error_function)
        {
            if (ann != NULL)
            {
                fann_set_train_error_function(ann,
					static_cast<fann_errorfunc_enum>(train_error_function));
            }
        }

        /* Method: get_quickprop_decay

           The decay is a small negative valued number which is the factor that the weights 
           should become smaller in each iteration during quickprop training. This is used 
           to make sure that the weights do not become too high during training.
           
           The default decay is -0.0001.
           
           See also:
   	        <set_quickprop_decay>, <fann_get_quickprop_decay>

           This function appears in FANN >= 1.2.0.
         */
        float get_quickprop_decay()
        {
            float quickprop_decay = 0.0f;
            if (ann != NULL)
            {
                quickprop_decay = fann_get_quickprop_decay(ann);
            }
            return quickprop_decay;
        }

        /* Method: set_quickprop_decay
           
           Sets the quickprop decay factor.
           
           See also:
   	        <get_quickprop_decay>, <fann_set_quickprop_decay>

           This function appears in FANN >= 1.2.0.
        */ 
        void set_quickprop_decay(float quickprop_decay)
        {
            if (ann != NULL)
            {
                fann_set_quickprop_decay(ann, quickprop_decay);
            }
        }

        /* Method: get_quickprop_mu

           The mu factor is used to increase and decrease the step-size during quickprop training. 
           The mu factor should always be above 1, since it would otherwise decrease the step-size 
           when it was suppose to increase it.
           
           The default mu factor is 1.75. 
           
           See also:
   	        <set_quickprop_mu>, <fann_get_quickprop_mu>

           This function appears in FANN >= 1.2.0.
        */ 
        float get_quickprop_mu()
        {
            float quickprop_mu = 0.0f;
            if (ann != NULL)
            {
                quickprop_mu = fann_get_quickprop_mu(ann);
            }
            return quickprop_mu;
        }

        /* Method: set_quickprop_mu

            Sets the quickprop mu factor.
           
           See also:
   	        <get_quickprop_mu>, <fann_set_quickprop_mu>

           This function appears in FANN >= 1.2.0.
        */ 
        void set_quickprop_mu(float quickprop_mu)
        {
            if (ann != NULL)
            {
                fann_set_quickprop_mu(ann, quickprop_mu);
            }
        }

        /* Method: get_rprop_increase_factor

           The increase factor is a value larger than 1, which is used to 
           increase the step-size during RPROP training.

           The default increase factor is 1.2.
           
           See also:
   	        <set_rprop_increase_factor>, <fann_get_rprop_increase_factor>

           This function appears in FANN >= 1.2.0.
        */ 
        float get_rprop_increase_factor()
        {
            float factor = 0.0f;
            if (ann != NULL)
            {
                factor = fann_get_rprop_increase_factor(ann);
            }
            return factor;
        }

        /* Method: set_rprop_increase_factor

           The increase factor used during RPROP training.

           See also:
   	        <get_rprop_increase_factor>, <fann_set_rprop_increase_factor>

           This function appears in FANN >= 1.2.0.
        */ 
        void set_rprop_increase_factor(float rprop_increase_factor)
        {
            if (ann != NULL)
            {
                fann_set_rprop_increase_factor(ann, rprop_increase_factor);
            }
        }

        /* Method: get_rprop_decrease_factor

           The decrease factor is a value smaller than 1, which is used to decrease the step-size during RPROP training.

           The default decrease factor is 0.5.

           See also:
            <set_rprop_decrease_factor>, <fann_get_rprop_decrease_factor>

           This function appears in FANN >= 1.2.0.
        */ 
        float get_rprop_decrease_factor()
        {
            float factor = 0.0f;
            if (ann != NULL)
            {
                factor = fann_get_rprop_decrease_factor(ann);
            }
            return factor;
        }

        /* Method: set_rprop_decrease_factor

           The decrease factor is a value smaller than 1, which is used to decrease the step-size during RPROP training.

           See also:
            <get_rprop_decrease_factor>, <fann_set_rprop_decrease_factor>

           This function appears in FANN >= 1.2.0.
        */
        void set_rprop_decrease_factor(float rprop_decrease_factor)
        {
            if (ann != NULL)
            {
                fann_set_rprop_decrease_factor(ann, rprop_decrease_factor);
            }
        }

        /* Method: get_rprop_delta_zero

           The initial step-size is a small positive number determining how small the initial step-size may be.

           The default value delta zero is 0.1.

           See also:
   	        <set_rprop_delta_zero>, <fann_get_rprop_delta_zero>
           	
           This function appears in FANN >= 2.1.0.
        */ 
        float get_rprop_delta_zero()
        {
            float delta = 0.0f;
            if (ann != NULL)
            {
                delta = fann_get_rprop_delta_zero(ann);
            }
            return delta;
        }

        /* Method: set_rprop_delta_zero

           The initial step-size is a small positive number determining how small the initial step-size may be.

           See also:
   	        <get_rprop_delta_zero>, <fann_set_rprop_delta_zero>
           	
           This function appears in FANN >= 2.1.0.
        */ 
        void set_rprop_delta_zero(float rprop_delta_zero)
        {
            if (ann != NULL)
            {
                fann_set_rprop_delta_zero(ann, rprop_delta_zero);
            }
        }
        /* Method: get_rprop_delta_min

           The minimum step-size is a small positive number determining how small the minimum step-size may be.

           The default value delta min is 0.0.

           See also:
   	        <set_rprop_delta_min>, <fann_get_rprop_delta_min>
           	
           This function appears in FANN >= 1.2.0.
        */ 
        float get_rprop_delta_min()
        {
            float delta = 0.0f;
            if (ann != NULL)
            {
                delta = fann_get_rprop_delta_min(ann);
            }
            return delta;
        }

        /* Method: set_rprop_delta_min

           The minimum step-size is a small positive number determining how small the minimum step-size may be.

           See also:
   	        <get_rprop_delta_min>, <fann_set_rprop_delta_min>
           	
           This function appears in FANN >= 1.2.0.
        */ 
        void set_rprop_delta_min(float rprop_delta_min)
        {
            if (ann != NULL)
            {
                fann_set_rprop_delta_min(ann, rprop_delta_min);
            }
        }

        /* Method: get_rprop_delta_max

           The maximum step-size is a positive number determining how large the maximum step-size may be.

           The default delta max is 50.0.

           See also:
   	        <set_rprop_delta_max>, <get_rprop_delta_min>, <fann_get_rprop_delta_max>

           This function appears in FANN >= 1.2.0.
        */ 
        float get_rprop_delta_max()
        {
            float delta = 0.0f;
            if (ann != NULL)
            {
                delta = fann_get_rprop_delta_max(ann);
            }
            return delta;
        }

        /* Method: set_rprop_delta_max

           The maximum step-size is a positive number determining how large the maximum step-size may be.

           See also:
   	        <get_rprop_delta_max>, <get_rprop_delta_min>, <fann_set_rprop_delta_max>

           This function appears in FANN >= 1.2.0.
        */
        void set_rprop_delta_max(float rprop_delta_max)
        {
            if (ann != NULL)
            {
                fann_set_rprop_delta_max(ann, rprop_delta_max);
            }
        }

        /* Method: get_sarprop_weight_decay_shift

           The sarprop weight decay shift.

           The default delta max is -6.644.

           See also:
   	        <set_sarprop_weight_decay_shift>, <fann get_sarprop_weight_decay_shift>

           This function appears in FANN >= 2.1.0.
        */ 
        float get_sarprop_weight_decay_shift()
        {
            float res = 0.0f;
            if (ann != NULL)
            {
                res = fann_get_rprop_delta_max(ann);
            }
            return res;
        }

        /* Method: set_sarprop_weight_decay_shift

           Set the sarprop weight decay shift.

	        This function appears in FANN >= 2.1.0.
           
	    See also:
   	        <get_sarprop_weight_decay_shift>, <fann_set_sarprop_weight_decay_shift>
        */ 
        void set_sarprop_weight_decay_shift(float sarprop_weight_decay_shift)
        {
            if (ann != NULL)
            {
                fann_set_sarprop_weight_decay_shift(ann, sarprop_weight_decay_shift);
            }
        }

        /* Method: get_sarprop_step_error_threshold_factor

           The sarprop step error threshold factor.

           The default delta max is 0.1.

           See also:
   	        <set_sarprop_step_error_threshold_factor>, <fann get_sarprop_step_error_threshold_factor>

           This function appears in FANN >= 2.1.0.
        */ 
        float get_sarprop_step_error_threshold_factor()
        {
            float res = 0.0f;
            if (ann != NULL)
            {
                res = fann_get_rprop_delta_max(ann);
            }
            return res;
        }

        /* Method: set_sarprop_step_error_threshold_factor

           Set the sarprop step error threshold factor.

	        This function appears in FANN >= 2.1.0.
           
	    See also:
   	        <get_sarprop_step_error_threshold_factor>, <fann_set_sarprop_step_error_threshold_factor>
        */ 
        void set_sarprop_step_error_threshold_factor(float sarprop_step_error_threshold_factor)
        {
            if (ann != NULL)
            {
                fann_set_sarprop_step_error_threshold_factor(ann, sarprop_step_error_threshold_factor);
            }
        }

        /* Method: get_sarprop_step_error_shift

           The get sarprop step error shift.

           The default delta max is 1.385.

           See also:
   	        <set_sarprop_step_error_shift>, <fann get_sarprop_step_error_shift>

           This function appears in FANN >= 2.1.0.
        */ 
        float get_sarprop_step_error_shift()
        {
            float res = 0.0f;
            if (ann != NULL)
            {
                res = fann_get_rprop_delta_max(ann);
            }
            return res;
        }

        /* Method: set_sarprop_step_error_shift

           Set the sarprop step error shift.

	        This function appears in FANN >= 2.1.0.
           
	    See also:
   	        <get_sarprop_step_error_shift>, <fann_set_sarprop_step_error_shift>
        */ 
        void set_sarprop_step_error_shift(float sarprop_step_error_shift)
        {
            if (ann != NULL)
            {
                fann_set_sarprop_step_error_shift(ann, sarprop_step_error_shift);
            }
        }
        
	/* Method: get_sarprop_temperature

           The sarprop weight decay shift.

           The default delta max is 0.015.

           See also:
   	        <set_sarprop_temperature>, <fann get_sarprop_temperature>

           This function appears in FANN >= 2.1.0.
        */ 
        float get_sarprop_temperature()
        {
            float res = 0.0f;
            if (ann != NULL)
            {
                res = fann_get_rprop_delta_max(ann);
            }
            return res;
        }

        /* Method: set_sarprop_temperature

           Set the sarprop_temperature.

	        This function appears in FANN >= 2.1.0.
           
	    See also:
   	        <get_sarprop_temperature>, <fann_set_sarprop_temperature>
        */ 
        void set_sarprop_temperature(float sarprop_temperature)
        {
            if (ann != NULL)
            {
                fann_set_sarprop_temperature(ann, sarprop_temperature);
            }
        }


        /* Method: get_num_input

           Get the number of input neurons.

	        This function appears in FANN >= 1.0.0.
        */ 
        unsigned int get_num_input()
        {
            unsigned int num_input = 0;
            if (ann != NULL)
            {
                num_input = fann_get_num_input(ann);
            }
            return num_input;
        }

        /* Method: get_num_output

           Get the number of output neurons.

	        This function appears in FANN >= 1.0.0.
        */ 
        unsigned int get_num_output()
        {
            unsigned int num_output = 0;
            if (ann != NULL)
            {
                num_output = fann_get_num_output(ann);
            }
            return num_output;
        }

        /* Method: get_total_neurons

           Get the total number of neurons in the entire network. This number does also include the 
	        bias neurons, so a 2-4-2 network has 2+4+2 +2(bias) = 10 neurons.

	        This function appears in FANN >= 1.0.0.
        */ 
        unsigned int get_total_neurons()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_total_neurons(ann);
        }

        /* Method: get_total_connections

           Get the total number of connections in the entire network.

	        This function appears in FANN >= 1.0.0.
        */ 
        unsigned int get_total_connections()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_total_connections(ann);
        }

#ifdef FIXEDFANN
        /* Method: get_decimal_point

	        Returns the position of the decimal point in the ann.

	        This function is only available when the ANN is in fixed point mode.

	        The decimal point is described in greater detail in the tutorial <Fixed Point Usage>.

	        See also:
		        <Fixed Point Usage>, <get_multiplier>, <save_to_fixed>,
                <training_data::save_train_to_fixed>, <fann_get_decimal_point>

	        This function appears in FANN >= 1.0.0.
        */ 
        unsigned int get_decimal_point()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_decimal_point(ann);
        }

        /* Method: get_multiplier

            Returns the multiplier that fix point data is multiplied with.

	        This function is only available when the ANN is in fixed point mode.

	        The multiplier is the used to convert between floating point and fixed point notation. 
	        A floating point number is multiplied with the multiplier in order to get the fixed point
	        number and visa versa.

	        The multiplier is described in greater detail in the tutorial <Fixed Point Usage>.

	        See also:
		        <Fixed Point Usage>, <get_decimal_point>, <save_to_fixed>,
                <training_data::save_train_to_fixed>, <fann_get_multiplier>

	        This function appears in FANN >= 1.0.0.
        */ 
        unsigned int get_multiplier()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_multiplier(ann);
        }
#endif /* FIXEDFANN */

        /*********************************************************************/

        /* Method: get_network_type

            Get the type of neural network it was created as.

	        Returns:
                The neural network type from enum <FANN::network_type_enum>

            See Also:
                <fann_get_network_type>

           This function appears in FANN >= 2.1.0
        */
        network_type_enum get_network_type()
        {
            fann_nettype_enum network_type = FANN_NETTYPE_LAYER;
            if (ann != NULL)
            {
                network_type = fann_get_network_type(ann);
            }
            return static_cast<network_type_enum>(network_type);
        }

        /* Method: get_connection_rate

            Get the connection rate used when the network was created

	        Returns:
                The connection rate

            See also:
                <fann_get_connection_rate>

           This function appears in FANN >= 2.1.0
        */
        float get_connection_rate()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_connection_rate(ann);
        }

        /* Method: get_num_layers

            Get the number of layers in the network

	        Returns:
		        The number of layers in the neural network

            See also:
                <fann_get_num_layers>

           This function appears in FANN >= 2.1.0
        */
        unsigned int get_num_layers()
        {
            if (ann == NULL)
            {
                return 0;
            }
            return fann_get_num_layers(ann);
        }

        /* Method: get_layer_array

            Get the number of neurons in each layer in the network.

            Bias is not included so the layers match the create methods.

            The layers array must be preallocated to at least
            sizeof(unsigned int) * get_num_layers() long.

            See also:
                <fann_get_layer_array>

           This function appears in FANN >= 2.1.0
        */
        void get_layer_array(unsigned int *layers)
        {
            if (ann != NULL)
            {
                fann_get_layer_array(ann, layers);
            }
        }

        /* Method: get_bias_array

            Get the number of bias in each layer in the network.

            The bias array must be preallocated to at least
            sizeof(unsigned int) * get_num_layers() long.

            See also:
                <fann_get_bias_array>

            This function appears in FANN >= 2.1.0
        */
        void get_bias_array(unsigned int *bias)
        {
            if (ann != NULL)
            {
                fann_get_bias_array(ann, bias);
            }
        }

        /* Method: get_connection_array

            Get the connections in the network.

            The connections array must be preallocated to at least
            sizeof(struct fann_connection) * get_total_connections() long.

            See also:
                <fann_get_connection_array>

           This function appears in FANN >= 2.1.0
        */
        void get_connection_array(connection *connections)
        {
            if (ann != NULL)
            {
                fann_get_connection_array(ann, connections);
            }
        }

        /* Method: set_weight_array

            Set connections in the network.

            Only the weights can be changed, connections and weights are ignored
            if they do not already exist in the network.

            The array must have sizeof(struct fann_connection) * num_connections size.

            See also:
                <fann_set_weight_array>

           This function appears in FANN >= 2.1.0
        */
        void set_weight_array(connection *connections, unsigned int num_connections)
        {
            if (ann != NULL)
            {
                fann_set_weight_array(ann, connections, num_connections);
            }
        }

        /* Method: set_weight

            Set a connection in the network.

            Only the weights can be changed. The connection/weight is
            ignored if it does not already exist in the network.

            See also:
                <fann_set_weight>

           This function appears in FANN >= 2.1.0
        */
        void set_weight(unsigned int from_neuron, unsigned int to_neuron, fann_type weight)
        {
            if (ann != NULL)
            {
                fann_set_weight(ann, from_neuron, to_neuron, weight);
            }
        }

        /*********************************************************************/

        /* Method: get_learning_momentum

           Get the learning momentum.
           
           The learning momentum can be used to speed up FANN::TRAIN_INCREMENTAL training.
           A too high momentum will however not benefit training. Setting momentum to 0 will
           be the same as not using the momentum parameter. The recommended value of this parameter
           is between 0.0 and 1.0.

           The default momentum is 0.
           
           See also:
           <set_learning_momentum>, <set_training_algorithm>

           This function appears in FANN >= 2.0.0.   	
         */ 
        float get_learning_momentum()
        {
            float learning_momentum = 0.0f;
            if (ann != NULL)
            {
                learning_momentum = fann_get_learning_momentum(ann);
            }
            return learning_momentum;
        }

        /* Method: set_learning_momentum

           Set the learning momentum.

           More info available in <get_learning_momentum>

           This function appears in FANN >= 2.0.0.   	
         */ 
        void set_learning_momentum(float learning_momentum)
        {
            if (ann != NULL)
            {
                fann_set_learning_momentum(ann, learning_momentum);
            }
        }

        /* Method: get_train_stop_function

           Returns the the stop function used during training.
           
           The stop function is described further in <FANN::stop_function_enum>
           
           The default stop function is FANN::STOPFUNC_MSE
           
           See also:
   	        <get_train_stop_function>, <get_bit_fail_limit>
              
           This function appears in FANN >= 2.0.0.
         */ 
        stop_function_enum get_train_stop_function()
        {
            enum fann_stopfunc_enum stopfunc = FANN_STOPFUNC_MSE;
            if (ann != NULL)
            {
                stopfunc = fann_get_train_stop_function(ann);
            }
            return static_cast<stop_function_enum>(stopfunc);
        }

        /* Method: set_train_stop_function

           Set the stop function used during training.

           The stop function is described further in <FANN::stop_function_enum>
           
           See also:
   	        <get_train_stop_function>
              
           This function appears in FANN >= 2.0.0.
         */ 
        void set_train_stop_function(stop_function_enum train_stop_function)
        {
            if (ann != NULL)
            {
                fann_set_train_stop_function(ann,
                    static_cast<enum fann_stopfunc_enum>(train_stop_function));
            }
        }

        /* Method: get_bit_fail_limit

           Returns the bit fail limit used during training.
           
           The bit fail limit is used during training when the <FANN::stop_function_enum> is set to FANN_STOPFUNC_BIT.

           The limit is the maximum accepted difference between the desired output and the actual output during
           training. Each output that diverges more than this limit is counted as an error bit.
           This difference is divided by two when dealing with symmetric activation functions,
           so that symmetric and not symmetric activation functions can use the same limit.
           
           The default bit fail limit is 0.35.
           
           See also:
   	        <set_bit_fail_limit>
           
           This function appears in FANN >= 2.0.0.
         */ 
        fann_type get_bit_fail_limit()
        {
            fann_type bit_fail_limit = 0.0f;

            if (ann != NULL)
            {
                bit_fail_limit = fann_get_bit_fail_limit(ann);
            }
            return bit_fail_limit;
        }

        /* Method: set_bit_fail_limit

           Set the bit fail limit used during training.
          
           See also:
   	        <get_bit_fail_limit>
           
           This function appears in FANN >= 2.0.0.
         */
        void set_bit_fail_limit(fann_type bit_fail_limit)
        {
            if (ann != NULL)
            {
                fann_set_bit_fail_limit(ann, bit_fail_limit);
            }
        }

        /* Method: get_bit_fail
        	
	        The number of fail bits; means the number of output neurons which differ more 
	        than the bit fail limit (see <get_bit_fail_limit>, <set_bit_fail_limit>). 
	        The bits are counted in all of the training data, so this number can be higher than
	        the number of training data.
        	
	        This value is reset by <reset_MSE> and updated by all the same functions which also
	        updates the MSE value (e.g. <test_data>, <train_epoch>)
        	
	        See also:
		        <FANN::stop_function_enum>, <get_MSE>

	        This function appears in FANN >= 2.0.0
        */
        unsigned int get_bit_fail()
        {
            unsigned int bit_fail = 0;
            if (ann != NULL)
            {
                bit_fail = fann_get_bit_fail(ann);
            }
            return bit_fail;
        }

        /*********************************************************************/

        /* Method: cascadetrain_on_data

           Trains on an entire dataset, for a period of time using the Cascade2 training algorithm.
           This algorithm adds neurons to the neural network while training, which means that it
           needs to start with an ANN without any hidden layers. The neural network should also use
           shortcut connections, so <create_shortcut> should be used to create the ANN like this:
           >net.create_shortcut(2, train_data.num_input_train_data(), train_data.num_output_train_data());
           
           This training uses the parameters set using the set_cascade_..., but it also uses another
           training algorithm as it's internal training algorithm. This algorithm can be set to either
           FANN::TRAIN_RPROP or FANN::TRAIN_QUICKPROP by <set_training_algorithm>, and the parameters 
           set for these training algorithms will also affect the cascade training.
           
           Parameters:
   		        data - The data, which should be used during training
   		        max_neuron - The maximum number of neurons to be added to neural network
   		        neurons_between_reports - The number of neurons between printing a status report to stdout.
   			        A value of zero means no reports should be printed.
   		        desired_error - The desired <fann_get_MSE> or <fann_get_bit_fail>, depending on which stop function
   			        is chosen by <fann_set_train_stop_function>.

	        Instead of printing out reports every neurons_between_reports, a callback function can be called 
	        (see <set_callback>).
        	
	        See also:
		        <train_on_data>, <cascadetrain_on_file>, <fann_cascadetrain_on_data>

	        This function appears in FANN >= 2.0.0. 
        */
        void cascadetrain_on_data(const training_data &data, unsigned int max_neurons,
            unsigned int neurons_between_reports, float desired_error)
        {
            if ((ann != NULL) && (data.train_data != NULL))
            {
                fann_cascadetrain_on_data(ann, data.train_data, max_neurons,
                    neurons_between_reports, desired_error);
            }
        }

        /* Method: cascadetrain_on_file
           
           Does the same as <cascadetrain_on_data>, but reads the training data directly from a file.
           
           See also:
   		        <fann_cascadetrain_on_data>, <fann_cascadetrain_on_file>

	        This function appears in FANN >= 2.0.0.
        */ 
        void cascadetrain_on_file(const std::string &filename, unsigned int max_neurons,
            unsigned int neurons_between_reports, float desired_error)
        {
            if (ann != NULL)
            {
                fann_cascadetrain_on_file(ann, filename.c_str(),
                    max_neurons, neurons_between_reports, desired_error);
            }
        }

        /* Method: get_cascade_output_change_fraction

           The cascade output change fraction is a number between 0 and 1 determining how large a fraction
           the <get_MSE> value should change within <get_cascade_output_stagnation_epochs> during
           training of the output connections, in order for the training not to stagnate. If the training 
           stagnates, the training of the output connections will be ended and new candidates will be prepared.
           
           This means:
           If the MSE does not change by a fraction of <get_cascade_output_change_fraction> during a 
           period of <get_cascade_output_stagnation_epochs>, the training of the output connections
           is stopped because the training has stagnated.

           If the cascade output change fraction is low, the output connections will be trained more and if the
           fraction is high they will be trained less.
           
           The default cascade output change fraction is 0.01, which is equalent to a 1% change in MSE.

           See also:
   		        <set_cascade_output_change_fraction>, <get_MSE>,
                <get_cascade_output_stagnation_epochs>, <fann_get_cascade_output_change_fraction>

	        This function appears in FANN >= 2.0.0.
         */
        float get_cascade_output_change_fraction()
        {
            float change_fraction = 0.0f;
            if (ann != NULL)
            {
                change_fraction = fann_get_cascade_output_change_fraction(ann);
            }
            return change_fraction;
        }

        /* Method: set_cascade_output_change_fraction

           Sets the cascade output change fraction.
           
           See also:
   		        <get_cascade_output_change_fraction>, <fann_set_cascade_output_change_fraction>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_output_change_fraction(float cascade_output_change_fraction)
        {
            if (ann != NULL)
            {
                fann_set_cascade_output_change_fraction(ann, cascade_output_change_fraction);
            }
        }

        /* Method: get_cascade_output_stagnation_epochs

           The number of cascade output stagnation epochs determines the number of epochs training is allowed to
           continue without changing the MSE by a fraction of <get_cascade_output_change_fraction>.
           
           See more info about this parameter in <get_cascade_output_change_fraction>.
           
           The default number of cascade output stagnation epochs is 12.

           See also:
   		        <set_cascade_output_stagnation_epochs>, <get_cascade_output_change_fraction>,
                <fann_get_cascade_output_stagnation_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_output_stagnation_epochs()
        {
            unsigned int stagnation_epochs = 0;
            if (ann != NULL)
            {
                stagnation_epochs = fann_get_cascade_output_stagnation_epochs(ann);
            }
            return stagnation_epochs;
        }

        /* Method: set_cascade_output_stagnation_epochs

           Sets the number of cascade output stagnation epochs.
           
           See also:
   		        <get_cascade_output_stagnation_epochs>, <fann_set_cascade_output_stagnation_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_output_stagnation_epochs(unsigned int cascade_output_stagnation_epochs)
        {
            if (ann != NULL)
            {
                fann_set_cascade_output_stagnation_epochs(ann, cascade_output_stagnation_epochs);
            }
        }

        /* Method: get_cascade_candidate_change_fraction

           The cascade candidate change fraction is a number between 0 and 1 determining how large a fraction
           the <get_MSE> value should change within <get_cascade_candidate_stagnation_epochs> during
           training of the candidate neurons, in order for the training not to stagnate. If the training 
           stagnates, the training of the candidate neurons will be ended and the best candidate will be selected.
           
           This means:
           If the MSE does not change by a fraction of <get_cascade_candidate_change_fraction> during a 
           period of <get_cascade_candidate_stagnation_epochs>, the training of the candidate neurons
           is stopped because the training has stagnated.

           If the cascade candidate change fraction is low, the candidate neurons will be trained more and if the
           fraction is high they will be trained less.
           
           The default cascade candidate change fraction is 0.01, which is equalent to a 1% change in MSE.

           See also:
   		        <set_cascade_candidate_change_fraction>, <get_MSE>,
                <get_cascade_candidate_stagnation_epochs>, <fann_get_cascade_candidate_change_fraction>

	        This function appears in FANN >= 2.0.0.
         */
        float get_cascade_candidate_change_fraction()
        {
            float change_fraction = 0.0f;
            if (ann != NULL)
            {
                change_fraction = fann_get_cascade_candidate_change_fraction(ann);
            }
            return change_fraction;
        }

        /* Method: set_cascade_candidate_change_fraction

           Sets the cascade candidate change fraction.
           
           See also:
   		        <get_cascade_candidate_change_fraction>,
                <fann_set_cascade_candidate_change_fraction>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_candidate_change_fraction(float cascade_candidate_change_fraction)
        {
            if (ann != NULL)
            {
                fann_set_cascade_candidate_change_fraction(ann, cascade_candidate_change_fraction);
            }
        }

        /* Method: get_cascade_candidate_stagnation_epochs

           The number of cascade candidate stagnation epochs determines the number of epochs training is allowed to
           continue without changing the MSE by a fraction of <get_cascade_candidate_change_fraction>.
           
           See more info about this parameter in <get_cascade_candidate_change_fraction>.

           The default number of cascade candidate stagnation epochs is 12.

           See also:
   		        <set_cascade_candidate_stagnation_epochs>, <get_cascade_candidate_change_fraction>,
                <fann_get_cascade_candidate_stagnation_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_candidate_stagnation_epochs()
        {
            unsigned int stagnation_epochs = 0;
            if (ann != NULL)
            {
                stagnation_epochs = fann_get_cascade_candidate_stagnation_epochs(ann);
            }
            return stagnation_epochs;
        }

        /* Method: set_cascade_candidate_stagnation_epochs

           Sets the number of cascade candidate stagnation epochs.
           
           See also:
   		        <get_cascade_candidate_stagnation_epochs>,
                <fann_set_cascade_candidate_stagnation_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_candidate_stagnation_epochs(unsigned int cascade_candidate_stagnation_epochs)
        {
            if (ann != NULL)
            {
                fann_set_cascade_candidate_stagnation_epochs(ann, cascade_candidate_stagnation_epochs);
            }
        }

        /* Method: get_cascade_weight_multiplier

           The weight multiplier is a parameter which is used to multiply the weights from the candidate neuron
           before adding the neuron to the neural network. This parameter is usually between 0 and 1, and is used
           to make the training a bit less aggressive.

           The default weight multiplier is 0.4

           See also:
   		        <set_cascade_weight_multiplier>, <fann_get_cascade_weight_multiplier>

	        This function appears in FANN >= 2.0.0.
         */
        fann_type get_cascade_weight_multiplier()
        {
            fann_type weight_multiplier = 0;
            if (ann != NULL)
            {
                weight_multiplier = fann_get_cascade_weight_multiplier(ann);
            }
            return weight_multiplier;
        }

        /* Method: set_cascade_weight_multiplier
           
           Sets the weight multiplier.
           
           See also:
   		        <get_cascade_weight_multiplier>, <fann_set_cascade_weight_multiplier>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_weight_multiplier(fann_type cascade_weight_multiplier)
        {
            if (ann != NULL)
            {
                fann_set_cascade_weight_multiplier(ann, cascade_weight_multiplier);
            }
        }

        /* Method: get_cascade_candidate_limit

           The candidate limit is a limit for how much the candidate neuron may be trained.
           The limit is a limit on the proportion between the MSE and candidate score.
           
           Set this to a lower value to avoid overfitting and to a higher if overfitting is
           not a problem.
           
           The default candidate limit is 1000.0

           See also:
   		        <set_cascade_candidate_limit>, <fann_get_cascade_candidate_limit>

	        This function appears in FANN >= 2.0.0.
         */
        fann_type get_cascade_candidate_limit()
        {
            fann_type candidate_limit = 0;
            if (ann != NULL)
            {
                candidate_limit = fann_get_cascade_candidate_limit(ann);
            }
            return candidate_limit;
        }

        /* Method: set_cascade_candidate_limit

           Sets the candidate limit.
          
           See also:
   		        <get_cascade_candidate_limit>, <fann_set_cascade_candidate_limit>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_candidate_limit(fann_type cascade_candidate_limit)
        {
            if (ann != NULL)
            {
                fann_set_cascade_candidate_limit(ann, cascade_candidate_limit);
            }
        }

        /* Method: get_cascade_max_out_epochs

           The maximum out epochs determines the maximum number of epochs the output connections
           may be trained after adding a new candidate neuron.
           
           The default max out epochs is 150

           See also:
   		        <set_cascade_max_out_epochs>, <fann_get_cascade_max_out_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_max_out_epochs()
        {
            unsigned int max_out_epochs = 0;
            if (ann != NULL)
            {
                max_out_epochs = fann_get_cascade_max_out_epochs(ann);
            }
            return max_out_epochs;
        }

        /* Method: set_cascade_max_out_epochs

           Sets the maximum out epochs.

           See also:
   		        <get_cascade_max_out_epochs>, <fann_set_cascade_max_out_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_max_out_epochs(unsigned int cascade_max_out_epochs)
        {
            if (ann != NULL)
            {
                fann_set_cascade_max_out_epochs(ann, cascade_max_out_epochs);
            }
        }

        /* Method: get_cascade_max_cand_epochs

           The maximum candidate epochs determines the maximum number of epochs the input 
           connections to the candidates may be trained before adding a new candidate neuron.
           
           The default max candidate epochs is 150

           See also:
   		        <set_cascade_max_cand_epochs>, <fann_get_cascade_max_cand_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_max_cand_epochs()
        {
            unsigned int max_cand_epochs = 0;
            if (ann != NULL)
            {
                max_cand_epochs = fann_get_cascade_max_cand_epochs(ann);
            }
            return max_cand_epochs;
        }

        /* Method: set_cascade_max_cand_epochs

           Sets the max candidate epochs.
          
           See also:
   		        <get_cascade_max_cand_epochs>, <fann_set_cascade_max_cand_epochs>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_max_cand_epochs(unsigned int cascade_max_cand_epochs)
        {
            if (ann != NULL)
            {
                fann_set_cascade_max_cand_epochs(ann, cascade_max_cand_epochs);
            }
        }

        /* Method: get_cascade_num_candidates

           The number of candidates used during training (calculated by multiplying <get_cascade_activation_functions_count>,
           <get_cascade_activation_steepnesses_count> and <get_cascade_num_candidate_groups>). 

           The actual candidates is defined by the <get_cascade_activation_functions> and 
           <get_cascade_activation_steepnesses> arrays. These arrays define the activation functions 
           and activation steepnesses used for the candidate neurons. If there are 2 activation functions
           in the activation function array and 3 steepnesses in the steepness array, then there will be 
           2x3=6 different candidates which will be trained. These 6 different candidates can be copied into
           several candidate groups, where the only difference between these groups is the initial weights.
           If the number of groups is set to 2, then the number of candidate neurons will be 2x3x2=12. The 
           number of candidate groups is defined by <set_cascade_num_candidate_groups>.

           The default number of candidates is 6x4x2 = 48

           See also:
   		        <get_cascade_activation_functions>, <get_cascade_activation_functions_count>, 
   		        <get_cascade_activation_steepnesses>, <get_cascade_activation_steepnesses_count>,
   		        <get_cascade_num_candidate_groups>, <fann_get_cascade_num_candidates>

	        This function appears in FANN >= 2.0.0.
         */ 
        unsigned int get_cascade_num_candidates()
        {
            unsigned int num_candidates = 0;
            if (ann != NULL)
            {
                num_candidates = fann_get_cascade_num_candidates(ann);
            }
            return num_candidates;
        }

        /* Method: get_cascade_activation_functions_count

           The number of activation functions in the <get_cascade_activation_functions> array.

           The default number of activation functions is 6.

           See also:
   		        <get_cascade_activation_functions>, <set_cascade_activation_functions>,
                <fann_get_cascade_activation_functions_count>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_activation_functions_count()
        {
            unsigned int activation_functions_count = 0;
            if (ann != NULL)
            {
                activation_functions_count = fann_get_cascade_activation_functions_count(ann);
            }
            return activation_functions_count;
        }

        /* Method: get_cascade_activation_functions

           The cascade activation functions array is an array of the different activation functions used by
           the candidates. 
           
           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this array.
           
           See also:
   		        <get_cascade_activation_functions_count>, <set_cascade_activation_functions>,
   		        <FANN::activation_function_enum>

	        This function appears in FANN >= 2.0.0.
         */
        activation_function_enum * get_cascade_activation_functions()
        {
            enum fann_activationfunc_enum *activation_functions = NULL;
            if (ann != NULL)
            {
                activation_functions = fann_get_cascade_activation_functions(ann);
            }
            return reinterpret_cast<activation_function_enum *>(activation_functions);
        }

        /* Method: set_cascade_activation_functions

           Sets the array of cascade candidate activation functions. The array must be just as long
           as defined by the count.

           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this array.

           See also:
   		        <get_cascade_activation_steepnesses_count>, <get_cascade_activation_steepnesses>,
                <fann_set_cascade_activation_functions>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_activation_functions(activation_function_enum *cascade_activation_functions,
            unsigned int cascade_activation_functions_count)
        {
            if (ann != NULL)
            {
                fann_set_cascade_activation_functions(ann,
                    reinterpret_cast<enum fann_activationfunc_enum *>(cascade_activation_functions),
                    cascade_activation_functions_count);
            }
        }

        /* Method: get_cascade_activation_steepnesses_count

           The number of activation steepnesses in the <get_cascade_activation_functions> array.

           The default number of activation steepnesses is 4.

           See also:
   		        <get_cascade_activation_steepnesses>, <set_cascade_activation_functions>,
                <fann_get_cascade_activation_steepnesses_count>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_activation_steepnesses_count()
        {
            unsigned int activation_steepness_count = 0;
            if (ann != NULL)
            {
                activation_steepness_count = fann_get_cascade_activation_steepnesses_count(ann);
            }
            return activation_steepness_count;
        }

        /* Method: get_cascade_activation_steepnesses

           The cascade activation steepnesses array is an array of the different activation functions used by
           the candidates.

           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this array.

           The default activation steepnesses is {0.25, 0.50, 0.75, 1.00}

           See also:
   		        <set_cascade_activation_steepnesses>, <get_cascade_activation_steepnesses_count>,
                <fann_get_cascade_activation_steepnesses>

	        This function appears in FANN >= 2.0.0.
         */
        fann_type *get_cascade_activation_steepnesses()
        {
            fann_type *activation_steepnesses = NULL;
            if (ann != NULL)
            {
                activation_steepnesses = fann_get_cascade_activation_steepnesses(ann);
            }
            return activation_steepnesses;
        }																

        /* Method: set_cascade_activation_steepnesses

           Sets the array of cascade candidate activation steepnesses. The array must be just as long
           as defined by the count.

           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this array.

           See also:
   		        <get_cascade_activation_steepnesses>, <get_cascade_activation_steepnesses_count>,
                <fann_set_cascade_activation_steepnesses>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_activation_steepnesses(fann_type *cascade_activation_steepnesses,
            unsigned int cascade_activation_steepnesses_count)
        {
            if (ann != NULL)
            {
                fann_set_cascade_activation_steepnesses(ann,
                    cascade_activation_steepnesses, cascade_activation_steepnesses_count);
            }
        }

        /* Method: get_cascade_num_candidate_groups

           The number of candidate groups is the number of groups of identical candidates which will be used
           during training.
           
           This number can be used to have more candidates without having to define new parameters for the candidates.
           
           See <get_cascade_num_candidates> for a description of which candidate neurons will be 
           generated by this parameter.
           
           The default number of candidate groups is 2

           See also:
   		        <set_cascade_num_candidate_groups>, <fann_get_cascade_num_candidate_groups>

	        This function appears in FANN >= 2.0.0.
         */
        unsigned int get_cascade_num_candidate_groups()
        {
            unsigned int num_candidate_groups = 0;
            if (ann != NULL)
            {
                num_candidate_groups = fann_get_cascade_num_candidate_groups(ann);
            }
            return num_candidate_groups;
        }

        /* Method: set_cascade_num_candidate_groups

           Sets the number of candidate groups.

           See also:
   		        <get_cascade_num_candidate_groups>, <fann_set_cascade_num_candidate_groups>

	        This function appears in FANN >= 2.0.0.
         */
        void set_cascade_num_candidate_groups(unsigned int cascade_num_candidate_groups)
        {
            if (ann != NULL)
            {
                fann_set_cascade_num_candidate_groups(ann, cascade_num_candidate_groups);
            }
        }

        /*********************************************************************/

#ifndef FIXEDFANN
        /* Method: scale_train

           Scale input and output data based on previously calculated parameters.

           See also:
   		        <descale_train>, <fann_scale_train>

	        This function appears in FANN >= 2.1.0.
         */
        void scale_train(training_data &data)
        {
            if (ann != NULL)
            {
                fann_scale_train(ann, data.train_data);
            }
        }

        /* Method: descale_train

           Descale input and output data based on previously calculated parameters.

           See also:
   		        <scale_train>, <fann_descale_train>

	        This function appears in FANN >= 2.1.0.
         */
        void descale_train(training_data &data)
        {
            if (ann != NULL)
            {
                fann_descale_train(ann, data.train_data);
            }
        }

        /* Method: set_input_scaling_params

           Calculate scaling parameters for future use based on training data.

           See also:
   		        <set_output_scaling_params>, <fann_set_input_scaling_params>

	        This function appears in FANN >= 2.1.0.
         */
        bool set_input_scaling_params(const training_data &data, float new_input_min, float new_input_max)
        {
            bool status = false;
            if (ann != NULL)
            {
                status = (fann_set_input_scaling_params(ann, data.train_data, new_input_min, new_input_max) != -1);
            }
            return status;
        }

        /* Method: set_output_scaling_params

           Calculate scaling parameters for future use based on training data.

           See also:
   		        <set_input_scaling_params>, <fann_set_output_scaling_params>

	        This function appears in FANN >= 2.1.0.
         */
        bool set_output_scaling_params(const training_data &data, float new_output_min, float new_output_max)
        {
            bool status = false;
            if (ann != NULL)
            {
                status = (fann_set_output_scaling_params(ann, data.train_data, new_output_min, new_output_max) != -1);
            }
            return status;
        }

        /* Method: set_scaling_params

           Calculate scaling parameters for future use based on training data.

           See also:
   		        <clear_scaling_params>, <fann_set_scaling_params>

	        This function appears in FANN >= 2.1.0.
         */
        bool set_scaling_params(const training_data &data,
	        float new_input_min, float new_input_max, float new_output_min, float new_output_max)
        {
            bool status = false;
            if (ann != NULL)
            {
                status = (fann_set_scaling_params(ann, data.train_data,
                    new_input_min, new_input_max, new_output_min, new_output_max) != -1);
            }
            return status;
        }

        /* Method: clear_scaling_params

           Clears scaling parameters.

           See also:
   		        <set_scaling_params>, <fann_clear_scaling_params>

	        This function appears in FANN >= 2.1.0.
         */
        bool clear_scaling_params()
        {
            bool status = false;
            if (ann != NULL)
            {
                status = (fann_clear_scaling_params(ann) != -1);
            }
            return status;
        }

        /* Method: scale_input

           Scale data in input vector before feed it to ann based on previously calculated parameters.

           See also:
   		        <descale_input>, <scale_output>, <fann_scale_input>

	        This function appears in FANN >= 2.1.0.
         */
        void scale_input(fann_type *input_vector)
        {
            if (ann != NULL)
            {
                fann_scale_input(ann, input_vector );
            }
        }

        /* Method: scale_output

           Scale data in output vector before feed it to ann based on previously calculated parameters.

           See also:
   		        <descale_output>, <scale_input>, <fann_scale_output>

	        This function appears in FANN >= 2.1.0.
         */
        void scale_output(fann_type *output_vector)
        {
            if (ann != NULL)
            {
                fann_scale_output(ann, output_vector );
            }
        }

        /* Method: descale_input

           Scale data in input vector after get it from ann based on previously calculated parameters.

           See also:
   		        <scale_input>, <descale_output>, <fann_descale_input>

	        This function appears in FANN >= 2.1.0.
         */
        void descale_input(fann_type *input_vector)
        {
            if (ann != NULL)
            {
                fann_descale_input(ann, input_vector );
            }
        }

        /* Method: descale_output

           Scale data in output vector after get it from ann based on previously calculated parameters.

           See also:
   		        <scale_output>, <descale_input>, <fann_descale_output>

	        This function appears in FANN >= 2.1.0.
         */
        void descale_output(fann_type *output_vector)
        {
            if (ann != NULL)
            {
                fann_descale_output(ann, output_vector );
            }
        }

#endif /* FIXEDFANN */

        /*********************************************************************/

        /* Method: set_error_log

           Change where errors are logged to.
           
           If log_file is NULL, no errors will be printed.
           
           If neural_net is empty i.e. ann is NULL, the default log will be set.
           The default log is the log used when creating a neural_net.
           This default log will also be the default for all new structs
           that are created.
           
           The default behavior is to log them to stderr.
           
           See also:
                <struct fann_error>, <fann_set_error_log>
           
           This function appears in FANN >= 1.1.0.   
         */ 
        void set_error_log(FILE *log_file)
        {
            fann_set_error_log(reinterpret_cast<struct fann_error *>(ann), log_file);
        }

        /* Method: get_errno

           Returns the last error number.
           
           See also:
            <fann_errno_enum>, <fann_reset_errno>, <fann_get_errno>
            
           This function appears in FANN >= 1.1.0.   
         */ 
        unsigned int get_errno()
        {
            return fann_get_errno(reinterpret_cast<struct fann_error *>(ann));
        }

        /* Method: reset_errno

           Resets the last error number.
           
           This function appears in FANN >= 1.1.0.   
         */ 
        void reset_errno()
        {
            fann_reset_errno(reinterpret_cast<struct fann_error *>(ann));
        }

        /* Method: reset_errstr

           Resets the last error string.

           This function appears in FANN >= 1.1.0.   
         */ 
        void reset_errstr()
        {
            fann_reset_errstr(reinterpret_cast<struct fann_error *>(ann));
        }

        /* Method: get_errstr

           Returns the last errstr.
          
           This function calls <fann_reset_errno> and <fann_reset_errstr>

           This function appears in FANN >= 1.1.0.   
         */ 
        std::string get_errstr()
        {
            return std::string(fann_get_errstr(reinterpret_cast<struct fann_error *>(ann)));
        }

        /* Method: print_error

           Prints the last error to stderr.

           This function appears in FANN >= 1.1.0.   
         */ 
        void print_error()
        {
            fann_print_error(reinterpret_cast<struct fann_error *>(ann));
        }

        /*********************************************************************/

    private:
        // Structure used by set_callback to hold information about a user callback
        typedef struct user_context_type
        {
            callback_type user_callback; // Pointer to user callback function
            void *user_data; // Arbitrary data pointer passed to the callback
            neural_net *net; // This pointer for the neural network
        } user_context;

        // Internal callback used to convert from pointers to class references
        static int FANN_API internal_callback(struct fann *ann, struct fann_train_data *train, 
            unsigned int max_epochs, unsigned int epochs_between_reports, float desired_error, unsigned int epochs)
        {
            user_context *user_data = static_cast<user_context *>(fann_get_user_data(ann));
            if (user_data != NULL)
            {
                FANN::training_data data;
                data.train_data = train;

                int result = (*user_data->user_callback)(*user_data->net,
                    data, max_epochs, epochs_between_reports, desired_error, epochs, user_data);

                data.train_data = NULL; // Prevent automatic cleanup
                return result;
            }
            else
            {
                return -1; // This should not occur except if out of memory
            }
        }
    protected:
        // Pointer the encapsulated fann neural net structure
        struct fann *ann;
    };

    /*************************************************************************/
}

#endif /* FANN_CPP_H_INCLUDED */
