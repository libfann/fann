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

#ifndef FANN_FANN_TRAINING_DATA_CPP_H
#define FANN_FANN_TRAINING_DATA_CPP_H

#include <stdarg.h>
#include <string>

namespace FANN {

    /* Section: FANN C++ Training Data
    */

    /* Class: training_data

    <training_data> is used to create and manipulate training data used by the <neural_net>

    Encapsulation of a training data set <struct fann_train_data> and
    associated C API functions.
    */
    class training_data {
    public:
        /* Constructor: training_data

            Default constructor creates an empty training data.
            Use <read_train_from_file>, <set_train_data> or <create_train_from_callback> to initialize.
        */
        training_data() : train_data(NULL) {
        }

        /* Constructor: training_data

            Copy constructor constructs a copy of the training data.
            Corresponds to the C API <fann_duplicate_train_data> function.
        */
        training_data(const training_data &data) {
            train_data = fann_duplicate_train_data(data.train_data);
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

        ~training_data() {
            destroy_train();
        }

        /* Method: destroy

            Destructs the training data. Called automatically by the destructor.

            See also:
                <~training_data>
        */
        void destroy_train() {
            if (train_data != NULL) {
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
        bool read_train_from_file(const std::string &filename) {
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
        bool save_train(const std::string &filename) {
            if (train_data == NULL) {
                return false;
            }
            if (fann_save_train(train_data, filename.c_str()) == -1) {
                return false;
            }
            return true;
        }

        /* Method: save_train_to_fixed

           Saves the training structure to a fixed point data file.

           This function is very useful for testing the quality of a fixed point network.

           Return:
           The function returns true on success and false on failure.

           See also:
   	        <save_train>, <fann_save_train_to_fixed>

           This function appears in FANN >= 1.0.0.
         */
        bool save_train_to_fixed(const std::string &filename, unsigned int decimal_point) {
            if (train_data == NULL) {
                return false;
            }
            if (fann_save_train_to_fixed(train_data, filename.c_str(), decimal_point) == -1) {
                return false;
            }
            return true;
        }

        /* Method: shuffle_train_data

           Shuffles training data, randomizing the order.
           This is recommended for incremental training, while it have no influence during batch training.

           This function appears in FANN >= 1.1.0.
         */
        void shuffle_train_data() {
            if (train_data != NULL) {
                fann_shuffle_train_data(train_data);
            }
        }

        /* Method: merge_train_data

           Merges the data into the data contained in the <training_data>.

           This function appears in FANN >= 1.1.0.
         */
        void merge_train_data(const training_data &data) {
            fann_train_data *new_data = fann_merge_train_data(train_data, data.train_data);
            if (new_data != NULL) {
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
        unsigned int length_train_data() {
            if (train_data == NULL) {
                return 0;
            }
            else {
                return fann_length_train_data(train_data);
            }
        }

        /* Method: num_input_train_data

           Returns the number of inputs in each of the training patterns in the <training_data>.

           See also:
           <num_output_train_data>, <length_train_data>, <fann_num_input_train_data>

           This function appears in FANN >= 2.0.0.
         */
        unsigned int num_input_train_data() {
            if (train_data == NULL) {
                return 0;
            }
            else {
                return fann_num_input_train_data(train_data);
            }
        }

        /* Method: num_output_train_data

           Returns the number of outputs in each of the training patterns in the <struct fann_train_data>.

           See also:
           <num_input_train_data>, <length_train_data>, <fann_num_output_train_data>

           This function appears in FANN >= 2.0.0.
         */
        unsigned int num_output_train_data() {
            if (train_data == NULL) {
                return 0;
            }
            else {
                return fann_num_output_train_data(train_data);
            }
        }

        /* Method: get_input
            Grant access to the encapsulated data since many situations
            and applications creates the data from sources other than files
            or uses the training data for testing and related functions

            Returns:
                A pointer to the array of input training data

            See also:
                <get_output>, <set_train_data>

           This function appears in FANN >= 2.0.0.
        */
        fann_type **get_input() {
            if (train_data == NULL) {
                return NULL;
            }
            else {
                return train_data->input;
            }
        }

        /* Method: get_output

            Grant access to the encapsulated data since many situations
            and applications creates the data from sources other than files
            or uses the training data for testing and related functions

            Returns:
                A pointer to the array of output training data

            See also:
                <get_input>, <set_train_data>

           This function appears in FANN >= 2.0.0.
        */
        fann_type **get_output() {
            if (train_data == NULL) {
                return NULL;
            }
            else {
                return train_data->output;
            }
        }

        /* Method: get_train_input
            Gets the training input data at the given position

            Returns:
                A pointer to the array of input training data at the given position

            See also:
                <get_train_output>, <set_train_data>

           This function appears in FANN >= 2.3.0.
        */
        fann_type *get_train_input(unsigned int position) {
            return fann_get_train_input(train_data, position);
        }

        /* Method: get_train_output
            Gets the training output data at the given position

            Returns:
                A pointer to the array of output training data at the given position

            See also:
                <get_train_input>

           This function appears in FANN >= 2.3.0.
        */
        fann_type *get_train_output(unsigned int position) {
            return fann_get_train_output(train_data, position);
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
                            unsigned int num_output, fann_type **output) {
            set_train_data(fann_create_train_pointer_array(num_data, num_input, input, num_output, output));
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
             input      - The set of inputs (an array with the dimension num_data*num_input)
             output     - The set of desired outputs (an array with the dimension num_data*num_output)

            See also:
                <get_input>, <get_output>
        */
        void set_train_data(unsigned int num_data,
                            unsigned int num_input, fann_type *input,
                            unsigned int num_output, fann_type *output) {
            set_train_data(fann_create_train_array(num_data, num_input, input, num_output, output));
        }

    private:
        /* Set the training data to the struct fann_training_data pointer.
            The struct has to be allocated with malloc to be compatible
            with fann_destroy. */
        void set_train_data(struct fann_train_data *data) {
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
                                        void (FANN_API *user_function)(unsigned int,
                                                                       unsigned int,
                                                                       unsigned int,
                                                                       fann_type *,
                                                                       fann_type *)) {
            destroy_train();
            train_data = fann_create_train_from_callback(num_data, num_input, num_output, user_function);
        }

#ifndef FIXEDFANN
        /* Function: get_min_input

           Get the minimum value of all in the input data

           This function appears in FANN >= 2.3.0
        */
        fann_type get_min_input() {
            return fann_get_min_train_input(train_data);
        }

        /* Function: get_max_input

           Get the maximum value of all in the input data

           This function appears in FANN >= 2.3.0
        */
        fann_type get_max_input() {
            return fann_get_max_train_input(train_data);
        }

        /* Function: get_min_output

           Get the minimum value of all in the output data

           This function appears in FANN >= 2.3.0
        */
        fann_type get_min_output() {
            return fann_get_min_train_output(train_data);
        }

        /* Function: get_max_output

           Get the maximum value of all in the output data

           This function appears in FANN >= 2.3.0
        */
        fann_type get_max_output() {
            return fann_get_max_train_output(train_data);
        }
#endif /* FIXEDFANN */

        /* Method: scale_input_train_data

           Scales the inputs in the training data to the specified range.

           A simplified scaling method, which is mostly useful in examples where it's known that all the
           data will be in one range and it should be transformed to another range.

           It is not recommended to use this on subsets of data as the complete input range might not be
           available in that subset.

           For more powerful scaling, please consider <neural_net::scale_train>

           See also:
   	        <scale_output_train_data>, <scale_train_data>, <fann_scale_input_train_data>

           This function appears in FANN >= 2.0.0.
         */
        void scale_input_train_data(fann_type new_min, fann_type new_max) {
            if (train_data != NULL) {
                fann_scale_input_train_data(train_data, new_min, new_max);
            }
        }

        /* Method: scale_output_train_data

           Scales the outputs in the training data to the specified range.

           A simplified scaling method, which is mostly useful in examples where it's known that all the
           data will be in one range and it should be transformed to another range.

           It is not recommended to use this on subsets of data as the complete input range might not be
           available in that subset.

           For more powerful scaling, please consider <neural_net::scale_train>

           See also:
   	        <scale_input_train_data>, <scale_train_data>, <fann_scale_output_train_data>

           This function appears in FANN >= 2.0.0.
         */
        void scale_output_train_data(fann_type new_min, fann_type new_max) {
            if (train_data != NULL) {
                fann_scale_output_train_data(train_data, new_min, new_max);
            }
        }

        /* Method: scale_train_data

           Scales the inputs and outputs in the training data to the specified range.

           A simplified scaling method, which is mostly useful in examples where it's known that all the
           data will be in one range and it should be transformed to another range.

           It is not recommended to use this on subsets of data as the complete input range might not be
           available in that subset.

           For more powerful scaling, please consider <neural_net::scale_train>

           See also:
   	        <scale_output_train_data>, <scale_input_train_data>, <fann_scale_train_data>

           This function appears in FANN >= 2.0.0.
         */
        void scale_train_data(fann_type new_min, fann_type new_max) {
            if (train_data != NULL) {
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
        void subset_train_data(unsigned int pos, unsigned int length) {
            if (train_data != NULL) {
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
        struct fann_train_data *train_data;
    };

}

#endif //FANN_FANN_TRAINING_DATA_CPP_H
