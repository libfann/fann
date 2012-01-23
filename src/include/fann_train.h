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

#ifndef __fann_train_h__
#define __fann_train_h__

/* Section: FANN Training 
 
 	There are many different ways of training neural networks and the FANN library supports
 	a number of different approaches. 
 	
 	Two fundementally different approaches are the most commonly used:
 	
 		Fixed topology training - The size and topology of the ANN is determined in advance
 			and the training alters the weights in order to minimize the difference between
 			the desired output values and the actual output values. This kind of training is 
 			supported by <fann_train_on_data>.
 			
 		Evolving topology training - The training start out with an empty ANN, only consisting
 			of input and output neurons. Hidden neurons and connections is the added during training,
 			in order to reach the same goal as for fixed topology training. This kind of training
 			is supported by <FANN Cascade Training>.
 */

/* Struct: struct fann_train_data
	Structure used to store data, for use with training.
	
	The data inside this structure should never be manipulated directly, but should use some 
	of the supplied functions in <Training Data Manipulation>.
	
	The training data structure is very usefull for storing data during training and testing of a
	neural network.
   
	See also:
	<fann_read_train_from_file>, <fann_train_on_data>, <fann_destroy_train>
*/
struct fann_train_data
{
	enum fann_errno_enum errno_f;
	FILE *error_log;
	char *errstr;

	unsigned int num_data;
	unsigned int num_input;
	unsigned int num_output;
	fann_type **input;
	fann_type **output;
};

/* Section: FANN Training */

/* Group: Training */

#ifndef FIXEDFANN
/* Function: fann_train

   Train one iteration with a set of inputs, and a set of desired outputs.
   This training is always incremental training (see <fann_train_enum>), since
   only one pattern is presented.
   
   Parameters:
   	ann - The neural network structure
   	input - an array of inputs. This array must be exactly <fann_get_num_input> long.
   	desired_output - an array of desired outputs. This array must be exactly <fann_get_num_output> long.
   	
   	See also:
   		<fann_train_on_data>, <fann_train_epoch>
   	
   	This function appears in FANN >= 1.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_train(struct fann *ann, fann_type * input,
									   fann_type * desired_output);

#endif	/* NOT FIXEDFANN */
	
/* Function: fann_test
   Test with a set of inputs, and a set of desired outputs.
   This operation updates the mean square error, but does not
   change the network in any way.
   
   See also:
   		<fann_test_data>, <fann_train>
   
   This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL fann_type * FANN_API fann_test(struct fann *ann, fann_type * input,
												 fann_type * desired_output);

/* Function: fann_get_MSE
   Reads the mean square error from the network.
   
   Reads the mean square error from the network. This value is calculated during 
   training or testing, and can therefore sometimes be a bit off if the weights 
   have been changed since the last calculation of the value.
   
   See also:
   	<fann_test_data>

	This function appears in FANN >= 1.1.0.
 */ 
FANN_EXTERNAL float FANN_API fann_get_MSE(struct fann *ann);

/* Function: fann_get_bit_fail
	
	The number of fail bits; means the number of output neurons which differ more 
	than the bit fail limit (see <fann_get_bit_fail_limit>, <fann_set_bit_fail_limit>). 
	The bits are counted in all of the training data, so this number can be higher than
	the number of training data.
	
	This value is reset by <fann_reset_MSE> and updated by all the same functions which also
	updates the MSE value (e.g. <fann_test_data>, <fann_train_epoch>)
	
	See also:
		<fann_stopfunc_enum>, <fann_get_MSE>

	This function appears in FANN >= 2.0.0
*/
FANN_EXTERNAL unsigned int FANN_API fann_get_bit_fail(struct fann *ann);

/* Function: fann_reset_MSE
   Resets the mean square error from the network.
   
   This function also resets the number of bits that fail.
   
   See also:
   	<fann_get_MSE>, <fann_get_bit_fail_limit>
   
    This function appears in FANN >= 1.1.0
 */ 
FANN_EXTERNAL void FANN_API fann_reset_MSE(struct fann *ann);

/* Group: Training Data Training */

#ifndef FIXEDFANN
	
/* Function: fann_train_on_data

   Trains on an entire dataset, for a period of time. 
   
   This training uses the training algorithm chosen by <fann_set_training_algorithm>,
   and the parameters set for these training algorithms.
   
   Parameters:
   		ann - The neural network
   		data - The data, which should be used during training
   		max_epochs - The maximum number of epochs the training should continue
   		epochs_between_reports - The number of epochs between printing a status report to stdout.
   			A value of zero means no reports should be printed.
   		desired_error - The desired <fann_get_MSE> or <fann_get_bit_fail>, depending on which stop function
   			is chosen by <fann_set_train_stop_function>.

	Instead of printing out reports every epochs_between_reports, a callback function can be called 
	(see <fann_set_callback>).
	
	See also:
		<fann_train_on_file>, <fann_train_epoch>, <Parameters>

	This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL void FANN_API fann_train_on_data(struct fann *ann, struct fann_train_data *data,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error);

/* Function: fann_train_on_file
   
   Does the same as <fann_train_on_data>, but reads the training data directly from a file.
   
   See also:
   		<fann_train_on_data>

	This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL void FANN_API fann_train_on_file(struct fann *ann, const char *filename,
											   unsigned int max_epochs,
											   unsigned int epochs_between_reports,
											   float desired_error);

/* Function: fann_train_epoch
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
		<fann_train_on_data>, <fann_test_data>
		
	This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL float FANN_API fann_train_epoch(struct fann *ann, struct fann_train_data *data);
#endif	/* NOT FIXEDFANN */

/* Function: fann_test_data
  
   Test a set of training data and calculates the MSE for the training data. 
   
   This function updates the MSE and the bit fail values.
   
   See also:
 	<fann_test>, <fann_get_MSE>, <fann_get_bit_fail>

	This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL float FANN_API fann_test_data(struct fann *ann, struct fann_train_data *data);

/* Group: Training Data Manipulation */

/* Function: fann_read_train_from_file
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
   	<fann_train_on_data>, <fann_destroy_train>, <fann_save_train>

    This function appears in FANN >= 1.0.0
*/ 
FANN_EXTERNAL struct fann_train_data *FANN_API fann_read_train_from_file(const char *filename);


/* Function: fann_create_train
   Creates an empty training data struct.
  
   See also:
     <fann_read_train_from_file>, <fann_train_on_data>, <fann_destroy_train>,
     <fann_save_train>

    This function appears in FANN >= 2.2.0
*/ 
FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train(unsigned int num_data, unsigned int num_input, unsigned int num_output);

/* Function: fann_create_train_from_callback
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
     <fann_read_train_from_file>, <fann_train_on_data>, <fann_destroy_train>,
     <fann_save_train>

    This function appears in FANN >= 2.1.0
*/ 
FANN_EXTERNAL struct fann_train_data * FANN_API fann_create_train_from_callback(unsigned int num_data,
                                          unsigned int num_input,
                                          unsigned int num_output,
                                          void (FANN_API *user_function)( unsigned int,
                                                                 unsigned int,
                                                                 unsigned int,
                                                                 fann_type * ,
                                                                 fann_type * ));

/* Function: fann_destroy_train
   Destructs the training data and properly deallocates all of the associated data.
   Be sure to call this function after finished using the training data.

    This function appears in FANN >= 1.0.0
 */ 
FANN_EXTERNAL void FANN_API fann_destroy_train(struct fann_train_data *train_data);


/* Function: fann_shuffle_train_data
   
   Shuffles training data, randomizing the order. 
   This is recommended for incremental training, while it have no influence during batch training.
   
   This function appears in FANN >= 1.1.0.
 */ 
FANN_EXTERNAL void FANN_API fann_shuffle_train_data(struct fann_train_data *train_data);

#ifndef FIXEDFANN
/* Function: fann_scale_train

   Scale input and output data based on previously calculated parameters.
   
   Parameters:
     ann      - ann for which were calculated trained parameters before
     data     - training data that needs to be scaled
     
   See also:
   	<fann_descale_train>, <fann_set_scaling_params>

    This function appears in FANN >= 2.1.0
*/
FANN_EXTERNAL void FANN_API fann_scale_train( struct fann *ann, struct fann_train_data *data );

/* Function: fann_descale_train

   Descale input and output data based on previously calculated parameters.
   
   Parameters:
     ann      - ann for which were calculated trained parameters before
     data     - training data that needs to be descaled
     
   See also:
   	<fann_scale_train>, <fann_set_scaling_params>

    This function appears in FANN >= 2.1.0
 */
FANN_EXTERNAL void FANN_API fann_descale_train( struct fann *ann, struct fann_train_data *data );

/* Function: fann_set_input_scaling_params

   Calculate input scaling parameters for future use based on training data.
   
   Parameters:
   	 ann           - ann for wgich parameters needs to be calculated
   	 data          - training data that will be used to calculate scaling parameters
   	 new_input_min - desired lower bound in input data after scaling (not strictly followed)
   	 new_input_max - desired upper bound in input data after scaling (not strictly followed)
   	 
   See also:
   	 <fann_set_output_scaling_params>

    This function appears in FANN >= 2.1.0
 */
FANN_EXTERNAL int FANN_API fann_set_input_scaling_params(
	struct fann *ann,
	const struct fann_train_data *data,
	float new_input_min,
	float new_input_max);

/* Function: fann_set_output_scaling_params

   Calculate output scaling parameters for future use based on training data.
   
   Parameters:
   	 ann            - ann for wgich parameters needs to be calculated
   	 data           - training data that will be used to calculate scaling parameters
   	 new_output_min - desired lower bound in input data after scaling (not strictly followed)
   	 new_output_max - desired upper bound in input data after scaling (not strictly followed)
   	 
   See also:
   	 <fann_set_input_scaling_params>

    This function appears in FANN >= 2.1.0
 */
FANN_EXTERNAL int FANN_API fann_set_output_scaling_params(
	struct fann *ann,
	const struct fann_train_data *data,
	float new_output_min,
	float new_output_max);

/* Function: fann_set_scaling_params

   Calculate input and output scaling parameters for future use based on training data.

   Parameters:
   	 ann            - ann for wgich parameters needs to be calculated
   	 data           - training data that will be used to calculate scaling parameters
   	 new_input_min  - desired lower bound in input data after scaling (not strictly followed)
   	 new_input_max  - desired upper bound in input data after scaling (not strictly followed)
   	 new_output_min - desired lower bound in input data after scaling (not strictly followed)
   	 new_output_max - desired upper bound in input data after scaling (not strictly followed)
   	 
   See also:
   	 <fann_set_input_scaling_params>, <fann_set_output_scaling_params>

    This function appears in FANN >= 2.1.0
 */
FANN_EXTERNAL int FANN_API fann_set_scaling_params(
	struct fann *ann,
	const struct fann_train_data *data,
	float new_input_min,
	float new_input_max,
	float new_output_min,
	float new_output_max);

/* Function: fann_clear_scaling_params

   Clears scaling parameters.
   
   Parameters:
     ann - ann for which to clear scaling parameters

    This function appears in FANN >= 2.1.0
 */
FANN_EXTERNAL int FANN_API fann_clear_scaling_params(struct fann *ann);

/* Function: fann_scale_input

   Scale data in input vector before feed it to ann based on previously calculated parameters.
   
   Parameters:
     ann          - for which scaling parameters were calculated
     input_vector - input vector that will be scaled
   
   See also:
     <fann_descale_input>, <fann_scale_output>

    This function appears in FANN >= 2.1.0
*/
FANN_EXTERNAL void FANN_API fann_scale_input( struct fann *ann, fann_type *input_vector );

/* Function: fann_scale_output

   Scale data in output vector before feed it to ann based on previously calculated parameters.
   
   Parameters:
     ann           - for which scaling parameters were calculated
     output_vector - output vector that will be scaled
   
   See also:
     <fann_descale_output>, <fann_scale_input>

    This function appears in FANN >= 2.1.0
 */
FANN_EXTERNAL void FANN_API fann_scale_output( struct fann *ann, fann_type *output_vector );

/* Function: fann_descale_input

   Scale data in input vector after get it from ann based on previously calculated parameters.
   
   Parameters:
     ann          - for which scaling parameters were calculated
     input_vector - input vector that will be descaled
   
   See also:
     <fann_scale_input>, <fann_descale_output>

    This function appears in FANN >= 2.1.0
 */
FANN_EXTERNAL void FANN_API fann_descale_input( struct fann *ann, fann_type *input_vector );

/* Function: fann_descale_output

   Scale data in output vector after get it from ann based on previously calculated parameters.
   
   Parameters:
     ann           - for which scaling parameters were calculated
     output_vector - output vector that will be descaled
   
   See also:
     <fann_scale_output>, <fann_descale_input>

    This function appears in FANN >= 2.1.0
 */
FANN_EXTERNAL void FANN_API fann_descale_output( struct fann *ann, fann_type *output_vector );

#endif

/* Function: fann_scale_input_train_data
   
   Scales the inputs in the training data to the specified range.

   See also:
   	<fann_scale_output_train_data>, <fann_scale_train_data>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_scale_input_train_data(struct fann_train_data *train_data,
														fann_type new_min, fann_type new_max);


/* Function: fann_scale_output_train_data
   
   Scales the outputs in the training data to the specified range.

   See also:
   	<fann_scale_input_train_data>, <fann_scale_train_data>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_scale_output_train_data(struct fann_train_data *train_data,
														 fann_type new_min, fann_type new_max);


/* Function: fann_scale_train_data
   
   Scales the inputs and outputs in the training data to the specified range.
   
   See also:
   	<fann_scale_output_train_data>, <fann_scale_input_train_data>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_scale_train_data(struct fann_train_data *train_data,
												  fann_type new_min, fann_type new_max);


/* Function: fann_merge_train_data
   
   Merges the data from *data1* and *data2* into a new <struct fann_train_data>.
   
   This function appears in FANN >= 1.1.0.
 */ 
FANN_EXTERNAL struct fann_train_data *FANN_API fann_merge_train_data(struct fann_train_data *data1,
																	 struct fann_train_data *data2);


/* Function: fann_duplicate_train_data
   
   Returns an exact copy of a <struct fann_train_data>.

   This function appears in FANN >= 1.1.0.
 */ 
FANN_EXTERNAL struct fann_train_data *FANN_API fann_duplicate_train_data(struct fann_train_data
																		 *data);
	
/* Function: fann_subset_train_data
   
   Returns an copy of a subset of the <struct fann_train_data>, starting at position *pos* 
   and *length* elements forward.
   
   >fann_subset_train_data(train_data, 0, fann_length_train_data(train_data))
   
   Will do the same as <fann_duplicate_train_data>.
   
   See also:
   	<fann_length_train_data>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL struct fann_train_data *FANN_API fann_subset_train_data(struct fann_train_data
																		 *data, unsigned int pos,
																		 unsigned int length);
	
/* Function: fann_length_train_data
   
   Returns the number of training patterns in the <struct fann_train_data>.

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_length_train_data(struct fann_train_data *data);
	
/* Function: fann_num_input_train_data
   
   Returns the number of inputs in each of the training patterns in the <struct fann_train_data>.
   
   See also:
   	<fann_num_train_data>, <fann_num_output_train_data>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_num_input_train_data(struct fann_train_data *data);
	
/* Function: fann_num_output_train_data
   
   Returns the number of outputs in each of the training patterns in the <struct fann_train_data>.
   
   See also:
   	<fann_num_train_data>, <fann_num_input_train_data>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_num_output_train_data(struct fann_train_data *data);
	
/* Function: fann_save_train
   
   Save the training structure to a file, with the format as specified in <fann_read_train_from_file>

   Return:
   The function returns 0 on success and -1 on failure.
      
   See also:
   	<fann_read_train_from_file>, <fann_save_train_to_fixed>
	
   This function appears in FANN >= 1.0.0.   	
 */ 
FANN_EXTERNAL int FANN_API fann_save_train(struct fann_train_data *data, const char *filename);


/* Function: fann_save_train_to_fixed
   
   Saves the training structure to a fixed point data file.
 
   This function is very usefull for testing the quality of a fixed point network.
   
   Return:
   The function returns 0 on success and -1 on failure.
   
   See also:
   	<fann_save_train>

   This function appears in FANN >= 1.0.0.   	
 */ 
FANN_EXTERNAL int FANN_API fann_save_train_to_fixed(struct fann_train_data *data, const char *filename,
													 unsigned int decimal_point);


/* Group: Parameters */

/* Function: fann_get_training_algorithm

   Return the training algorithm as described by <fann_train_enum>. This training algorithm
   is used by <fann_train_on_data> and associated functions.
   
   Note that this algorithm is also used during <fann_cascadetrain_on_data>, although only
   FANN_TRAIN_RPROP and FANN_TRAIN_QUICKPROP is allowed during cascade training.
   
   The default training algorithm is FANN_TRAIN_RPROP.
   
   See also:
    <fann_set_training_algorithm>, <fann_train_enum>

   This function appears in FANN >= 1.0.0.   	
 */ 
FANN_EXTERNAL enum fann_train_enum FANN_API fann_get_training_algorithm(struct fann *ann);


/* Function: fann_set_training_algorithm

   Set the training algorithm.
   
   More info available in <fann_get_training_algorithm>

   This function appears in FANN >= 1.0.0.   	
 */ 
FANN_EXTERNAL void FANN_API fann_set_training_algorithm(struct fann *ann,
														enum fann_train_enum training_algorithm);


/* Function: fann_get_learning_rate

   Return the learning rate.
   
   The learning rate is used to determine how aggressive training should be for some of the
   training algorithms (FANN_TRAIN_INCREMENTAL, FANN_TRAIN_BATCH, FANN_TRAIN_QUICKPROP).
   Do however note that it is not used in FANN_TRAIN_RPROP.
   
   The default learning rate is 0.7.
   
   See also:
   	<fann_set_learning_rate>, <fann_set_training_algorithm>
   
   This function appears in FANN >= 1.0.0.   	
 */ 
FANN_EXTERNAL float FANN_API fann_get_learning_rate(struct fann *ann);


/* Function: fann_set_learning_rate

   Set the learning rate.
   
   More info available in <fann_get_learning_rate>

   This function appears in FANN >= 1.0.0.   	
 */ 
FANN_EXTERNAL void FANN_API fann_set_learning_rate(struct fann *ann, float learning_rate);

/* Function: fann_get_learning_momentum

   Get the learning momentum.
   
   The learning momentum can be used to speed up FANN_TRAIN_INCREMENTAL training.
   A too high momentum will however not benefit training. Setting momentum to 0 will
   be the same as not using the momentum parameter. The recommended value of this parameter
   is between 0.0 and 1.0.

   The default momentum is 0.
   
   See also:
   <fann_set_learning_momentum>, <fann_set_training_algorithm>

   This function appears in FANN >= 2.0.0.   	
 */ 
FANN_EXTERNAL float FANN_API fann_get_learning_momentum(struct fann *ann);


/* Function: fann_set_learning_momentum

   Set the learning momentum.

   More info available in <fann_get_learning_momentum>

   This function appears in FANN >= 2.0.0.   	
 */ 
FANN_EXTERNAL void FANN_API fann_set_learning_momentum(struct fann *ann, float learning_momentum);


/* Function: fann_get_activation_function

   Get the activation function for neuron number *neuron* in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to get activation functions for the neurons in the input layer.
   
   Information about the individual activation functions is available at <fann_activationfunc_enum>.

   Returns:
    The activation function for the neuron or -1 if the neuron is not defined in the neural network.
   
   See also:
   	<fann_set_activation_function_layer>, <fann_set_activation_function_hidden>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness>,
    <fann_set_activation_function>

   This function appears in FANN >= 2.1.0
 */ 
FANN_EXTERNAL enum fann_activationfunc_enum FANN_API fann_get_activation_function(struct fann *ann,
																int layer,
																int neuron);

/* Function: fann_set_activation_function

   Set the activation function for neuron number *neuron* in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to set activation functions for the neurons in the input layer.
   
   When choosing an activation function it is important to note that the activation 
   functions have different range. FANN_SIGMOID is e.g. in the 0 - 1 range while 
   FANN_SIGMOID_SYMMETRIC is in the -1 - 1 range and FANN_LINEAR is unbound.
   
   Information about the individual activation functions is available at <fann_activationfunc_enum>.
   
   The default activation function is FANN_SIGMOID_STEPWISE.
   
   See also:
   	<fann_set_activation_function_layer>, <fann_set_activation_function_hidden>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness>,
    <fann_get_activation_function>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function,
																int layer,
																int neuron);

/* Function: fann_set_activation_function_layer

   Set the activation function for all the neurons in the layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to set activation functions for the neurons in the input layer.

   See also:
   	<fann_set_activation_function>, <fann_set_activation_function_hidden>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness_layer>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function_layer(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function,
																int layer);

/* Function: fann_set_activation_function_hidden

   Set the activation function for all of the hidden layers.

   See also:
   	<fann_set_activation_function>, <fann_set_activation_function_layer>,
   	<fann_set_activation_function_output>, <fann_set_activation_steepness_hidden>

   This function appears in FANN >= 1.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function_hidden(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function);


/* Function: fann_set_activation_function_output

   Set the activation function for the output layer.

   See also:
   	<fann_set_activation_function>, <fann_set_activation_function_layer>,
   	<fann_set_activation_function_hidden>, <fann_set_activation_steepness_output>

   This function appears in FANN >= 1.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_function_output(struct fann *ann,
																enum fann_activationfunc_enum
																activation_function);

/* Function: fann_get_activation_steepness

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
   	<fann_set_activation_steepness_layer>, <fann_set_activation_steepness_hidden>,
   	<fann_set_activation_steepness_output>, <fann_set_activation_function>,
    <fann_set_activation_steepness>

   This function appears in FANN >= 2.1.0
 */ 
FANN_EXTERNAL fann_type FANN_API fann_get_activation_steepness(struct fann *ann,
																int layer,
																int neuron);

/* Function: fann_set_activation_steepness

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
   	<fann_set_activation_steepness_layer>, <fann_set_activation_steepness_hidden>,
   	<fann_set_activation_steepness_output>, <fann_set_activation_function>,
    <fann_get_activation_steepness>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness(struct fann *ann,
																fann_type steepness,
																int layer,
																int neuron);

/* Function: fann_set_activation_steepness_layer

   Set the activation steepness all of the neurons in layer number *layer*, 
   counting the input layer as layer 0. 
   
   It is not possible to set activation steepness for the neurons in the input layer.
   
   See also:
   	<fann_set_activation_steepness>, <fann_set_activation_steepness_hidden>,
   	<fann_set_activation_steepness_output>, <fann_set_activation_function_layer>

   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness_layer(struct fann *ann,
																fann_type steepness,
																int layer);

/* Function: fann_set_activation_steepness_hidden

   Set the steepness of the activation steepness in all of the hidden layers.

   See also:
   	<fann_set_activation_steepness>, <fann_set_activation_steepness_layer>,
   	<fann_set_activation_steepness_output>, <fann_set_activation_function_hidden>

   This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness_hidden(struct fann *ann,
																 fann_type steepness);


/* Function: fann_set_activation_steepness_output

   Set the steepness of the activation steepness in the output layer.

   See also:
   	<fann_set_activation_steepness>, <fann_set_activation_steepness_layer>,
   	<fann_set_activation_steepness_hidden>, <fann_set_activation_function_output>

   This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_activation_steepness_output(struct fann *ann,
																 fann_type steepness);


/* Function: fann_get_train_error_function

   Returns the error function used during training.

   The error functions is described further in <fann_errorfunc_enum>
   
   The default error function is FANN_ERRORFUNC_TANH
   
   See also:
   	<fann_set_train_error_function>
      
   This function appears in FANN >= 1.2.0.
  */ 
FANN_EXTERNAL enum fann_errorfunc_enum FANN_API fann_get_train_error_function(struct fann *ann);


/* Function: fann_set_train_error_function

   Set the error function used during training.
   
   The error functions is described further in <fann_errorfunc_enum>
   
   See also:
   	<fann_get_train_error_function>
      
   This function appears in FANN >= 1.2.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_train_error_function(struct fann *ann,
														  enum fann_errorfunc_enum 
														  train_error_function);


/* Function: fann_get_train_stop_function

   Returns the the stop function used during training.
   
   The stop function is described further in <fann_stopfunc_enum>
   
   The default stop function is FANN_STOPFUNC_MSE
   
   See also:
   	<fann_get_train_stop_function>, <fann_get_bit_fail_limit>
      
   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL enum fann_stopfunc_enum FANN_API fann_get_train_stop_function(struct fann *ann);


/* Function: fann_set_train_stop_function

   Set the stop function used during training.

   Returns the the stop function used during training.
   
   The stop function is described further in <fann_stopfunc_enum>
   
   See also:
   	<fann_get_train_stop_function>
      
   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_train_stop_function(struct fann *ann,
														 enum fann_stopfunc_enum train_stop_function);


/* Function: fann_get_bit_fail_limit

   Returns the bit fail limit used during training.
   
   The bit fail limit is used during training where the <fann_stopfunc_enum> is set to FANN_STOPFUNC_BIT.

   The limit is the maximum accepted difference between the desired output and the actual output during
   training. Each output that diverges more than this limit is counted as an error bit.
   This difference is divided by two when dealing with symmetric activation functions,
   so that symmetric and not symmetric activation functions can use the same limit.
   
   The default bit fail limit is 0.35.
   
   See also:
   	<fann_set_bit_fail_limit>
   
   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL fann_type FANN_API fann_get_bit_fail_limit(struct fann *ann);

/* Function: fann_set_bit_fail_limit

   Set the bit fail limit used during training.
  
   See also:
   	<fann_get_bit_fail_limit>
   
   This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL void FANN_API fann_set_bit_fail_limit(struct fann *ann, fann_type bit_fail_limit);

/* Function: fann_set_callback
   
   Sets the callback function for use during training.
 	
   See <fann_callback_type> for more information about the callback function.
   
   The default callback function simply prints out some status information.

   This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_callback(struct fann *ann, fann_callback_type callback);

/* Function: fann_get_quickprop_decay

   The decay is a small negative valued number which is the factor that the weights 
   should become smaller in each iteration during quickprop training. This is used 
   to make sure that the weights do not become too high during training.
   
   The default decay is -0.0001.
   
   See also:
   	<fann_set_quickprop_decay>

   This function appears in FANN >= 1.2.0.
 */
FANN_EXTERNAL float FANN_API fann_get_quickprop_decay(struct fann *ann);


/* Function: fann_set_quickprop_decay
   
   Sets the quickprop decay factor.
   
   See also:
   	<fann_get_quickprop_decay>

   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL void FANN_API fann_set_quickprop_decay(struct fann *ann, float quickprop_decay);


/* Function: fann_get_quickprop_mu

   The mu factor is used to increase and decrease the step-size during quickprop training. 
   The mu factor should always be above 1, since it would otherwise decrease the step-size 
   when it was suppose to increase it.
   
   The default mu factor is 1.75. 
   
   See also:
   	<fann_set_quickprop_mu>

   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL float FANN_API fann_get_quickprop_mu(struct fann *ann);


/* Function: fann_set_quickprop_mu

    Sets the quickprop mu factor.
   
   See also:
   	<fann_get_quickprop_mu>

   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL void FANN_API fann_set_quickprop_mu(struct fann *ann, float quickprop_mu);


/* Function: fann_get_rprop_increase_factor

   The increase factor is a value larger than 1, which is used to 
   increase the step-size during RPROP training.

   The default increase factor is 1.2.
   
   See also:
   	<fann_set_rprop_increase_factor>

   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL float FANN_API fann_get_rprop_increase_factor(struct fann *ann);


/* Function: fann_set_rprop_increase_factor

   The increase factor used during RPROP training.

   See also:
   	<fann_get_rprop_increase_factor>

   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL void FANN_API fann_set_rprop_increase_factor(struct fann *ann,
														   float rprop_increase_factor);


/* Function: fann_get_rprop_decrease_factor

   The decrease factor is a value smaller than 1, which is used to decrease the step-size during RPROP training.

   The default decrease factor is 0.5.

   See also:
    <fann_set_rprop_decrease_factor>

   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL float FANN_API fann_get_rprop_decrease_factor(struct fann *ann);


/* Function: fann_set_rprop_decrease_factor

   The decrease factor is a value smaller than 1, which is used to decrease the step-size during RPROP training.

   See also:
    <fann_get_rprop_decrease_factor>

   This function appears in FANN >= 1.2.0.
*/
FANN_EXTERNAL void FANN_API fann_set_rprop_decrease_factor(struct fann *ann,
														   float rprop_decrease_factor);


/* Function: fann_get_rprop_delta_min

   The minimum step-size is a small positive number determining how small the minimum step-size may be.

   The default value delta min is 0.0.

   See also:
   	<fann_set_rprop_delta_min>
   	
   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL float FANN_API fann_get_rprop_delta_min(struct fann *ann);


/* Function: fann_set_rprop_delta_min

   The minimum step-size is a small positive number determining how small the minimum step-size may be.

   See also:
   	<fann_get_rprop_delta_min>
   	
   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL void FANN_API fann_set_rprop_delta_min(struct fann *ann, float rprop_delta_min);


/* Function: fann_get_rprop_delta_max

   The maximum step-size is a positive number determining how large the maximum step-size may be.

   The default delta max is 50.0.

   See also:
   	<fann_set_rprop_delta_max>, <fann_get_rprop_delta_min>

   This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL float FANN_API fann_get_rprop_delta_max(struct fann *ann);


/* Function: fann_set_rprop_delta_max

   The maximum step-size is a positive number determining how large the maximum step-size may be.

   See also:
   	<fann_get_rprop_delta_max>, <fann_get_rprop_delta_min>

   This function appears in FANN >= 1.2.0.
*/
FANN_EXTERNAL void FANN_API fann_set_rprop_delta_max(struct fann *ann, float rprop_delta_max);

/* Function: fann_get_rprop_delta_zero

   The initial step-size is a positive number determining the initial step size.

   The default delta zero is 0.1.

   See also:
   	<fann_set_rprop_delta_zero>, <fann_get_rprop_delta_min>, <fann_get_rprop_delta_max>

   This function appears in FANN >= 2.1.0.
*/ 
FANN_EXTERNAL float FANN_API fann_get_rprop_delta_zero(struct fann *ann);


/* Function: fann_set_rprop_delta_zero

   The initial step-size is a positive number determining the initial step size.

   See also:
   	<fann_get_rprop_delta_zero>, <fann_get_rprop_delta_zero>

   This function appears in FANN >= 2.1.0.
*/
FANN_EXTERNAL void FANN_API fann_set_rprop_delta_zero(struct fann *ann, float rprop_delta_max);

/* Method: fann_get_sarprop_weight_decay_shift

   The sarprop weight decay shift.

   The default delta max is -6.644.

   See also:
   <fann fann_set_sarprop_weight_decay_shift>

   This function appears in FANN >= 2.1.0.
   */ 
FANN_EXTERNAL float FANN_API fann_get_sarprop_weight_decay_shift(struct fann *ann);

/* Method: fann_set_sarprop_weight_decay_shift

   Set the sarprop weight decay shift.

   This function appears in FANN >= 2.1.0.

   See also:
   <fann_set_sarprop_weight_decay_shift>
   */ 
FANN_EXTERNAL void FANN_API fann_set_sarprop_weight_decay_shift(struct fann *ann, float sarprop_weight_decay_shift);

/* Method: fann_get_sarprop_step_error_threshold_factor

   The sarprop step error threshold factor.

   The default delta max is 0.1.

   See also:
   <fann fann_get_sarprop_step_error_threshold_factor>

   This function appears in FANN >= 2.1.0.
   */ 
FANN_EXTERNAL float FANN_API fann_get_sarprop_step_error_threshold_factor(struct fann *ann);

/* Method: fann_set_sarprop_step_error_threshold_factor

   Set the sarprop step error threshold factor.

   This function appears in FANN >= 2.1.0.

   See also:
   <fann_get_sarprop_step_error_threshold_factor>
   */ 
FANN_EXTERNAL void FANN_API fann_set_sarprop_step_error_threshold_factor(struct fann *ann, float sarprop_step_error_threshold_factor);

/* Method: fann_get_sarprop_step_error_shift

   The get sarprop step error shift.

   The default delta max is 1.385.

   See also:
   <fann_set_sarprop_step_error_shift>

   This function appears in FANN >= 2.1.0.
   */ 
FANN_EXTERNAL float FANN_API fann_get_sarprop_step_error_shift(struct fann *ann);

/* Method: fann_set_sarprop_step_error_shift

   Set the sarprop step error shift.

   This function appears in FANN >= 2.1.0.

   See also:
   <fann_get_sarprop_step_error_shift>
   */ 
FANN_EXTERNAL void FANN_API fann_set_sarprop_step_error_shift(struct fann *ann, float sarprop_step_error_shift);

/* Method: fann_get_sarprop_temperature

   The sarprop weight decay shift.

   The default delta max is 0.015.

   See also:
   <fann_set_sarprop_temperature>

   This function appears in FANN >= 2.1.0.
   */ 
FANN_EXTERNAL float FANN_API fann_get_sarprop_temperature(struct fann *ann);

/* Method: fann_set_sarprop_temperature

   Set the sarprop_temperature.

   This function appears in FANN >= 2.1.0.

   See also:
   <fann_get_sarprop_temperature>
   */ 
FANN_EXTERNAL void FANN_API fann_set_sarprop_temperature(struct fann *ann, float sarprop_temperature);

#endif
