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

#ifndef __fann_cascade_h__
#define __fann_cascade_h__

/* Section: FANN Cascade Training
   Cascade training differs from ordinary training in the sense that it starts with an empty neural network
   and then adds neurons one by one, while it trains the neural network. The main benefit of this approach
   is that you do not have to guess the number of hidden layers and neurons prior to training, but cascade 
   training has also proved better at solving some problems.
   
   The basic idea of cascade training is that a number of candidate neurons are trained separate from the 
   real network, then the most promising of these candidate neurons is inserted into the neural network. 
   Then the output connections are trained and new candidate neurons are prepared. The candidate neurons are 
   created as shortcut connected neurons in a new hidden layer, which means that the final neural network
   will consist of a number of hidden layers with one shortcut connected neuron in each.
*/

/* Group: Cascade Training */

#ifndef FIXEDFANN
/* Function: fann_cascadetrain_on_data

   Trains on an entire dataset, for a period of time using the Cascade2 training algorithm.
   This algorithm adds neurons to the neural network while training, which means that it
   needs to start with an ANN without any hidden layers. The neural network should also use
   shortcut connections, so <fann_create_shortcut> should be used to create the ANN like this:
   >struct fann *ann = fann_create_shortcut(2, fann_num_input_train_data(train_data), fann_num_output_train_data(train_data));
   
   This training uses the parameters set using the fann_set_cascade_..., but it also uses another
   training algorithm as it's internal training algorithm. This algorithm can be set to either
   FANN_TRAIN_RPROP or FANN_TRAIN_QUICKPROP by <fann_set_training_algorithm>, and the parameters 
   set for these training algorithms will also affect the cascade training.
   
   Parameters:
   		ann - The neural network
   		data - The data, which should be used during training
   		max_neuron - The maximum number of neurons to be added to neural network
   		neurons_between_reports - The number of neurons between printing a status report to stdout.
   			A value of zero means no reports should be printed.
   		desired_error - The desired <fann_get_MSE> or <fann_get_bit_fail>, depending on which stop function
   			is chosen by <fann_set_train_stop_function>.

	Instead of printing out reports every neurons_between_reports, a callback function can be called 
	(see <fann_set_callback>).
	
	See also:
		<fann_train_on_data>, <fann_cascadetrain_on_file>, <Parameters>

	This function appears in FANN >= 2.0.0. 
*/
FANN_EXTERNAL void FANN_API fann_cascadetrain_on_data(struct fann *ann,
													  struct fann_train_data *data,
													  unsigned int max_neurons,
													  unsigned int neurons_between_reports,
													  float desired_error);

/* Function: fann_cascadetrain_on_file
   
   Does the same as <fann_cascadetrain_on_data>, but reads the training data directly from a file.
   
   See also:
   		<fann_cascadetrain_on_data>

	This function appears in FANN >= 2.0.0.
*/ 
FANN_EXTERNAL void FANN_API fann_cascadetrain_on_file(struct fann *ann, const char *filename,
													  unsigned int max_neurons,
													  unsigned int neurons_between_reports,
													  float desired_error);

/* Group: Parameters */
													  
/* Function: fann_get_cascade_output_change_fraction

   The cascade output change fraction is a number between 0 and 1 determining how large a fraction
   the <fann_get_MSE> value should change within <fann_get_cascade_output_stagnation_epochs> during
   training of the output connections, in order for the training not to stagnate. If the training 
   stagnates, the training of the output connections will be ended and new candidates will be prepared.
   
   This means:
   If the MSE does not change by a fraction of <fann_get_cascade_output_change_fraction> during a 
   period of <fann_get_cascade_output_stagnation_epochs>, the training of the output connections
   is stopped because the training has stagnated.

   If the cascade output change fraction is low, the output connections will be trained more and if the
   fraction is high they will be trained less.
   
   The default cascade output change fraction is 0.01, which is equivalent to a 1% change in MSE.

   See also:
   		<fann_set_cascade_output_change_fraction>, <fann_get_MSE>, <fann_get_cascade_output_stagnation_epochs>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL float FANN_API fann_get_cascade_output_change_fraction(struct fann *ann);


/* Function: fann_set_cascade_output_change_fraction

   Sets the cascade output change fraction.
   
   See also:
   		<fann_get_cascade_output_change_fraction>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_output_change_fraction(struct fann *ann, 
															 float cascade_output_change_fraction);

/* Function: fann_get_cascade_output_stagnation_epochs

   The number of cascade output stagnation epochs determines the number of epochs training is allowed to
   continue without changing the MSE by a fraction of <fann_get_cascade_output_change_fraction>.
   
   See more info about this parameter in <fann_get_cascade_output_change_fraction>.
   
   The default number of cascade output stagnation epochs is 12.

   See also:
   		<fann_set_cascade_output_stagnation_epochs>, <fann_get_cascade_output_change_fraction>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_output_stagnation_epochs(struct fann *ann);


/* Function: fann_set_cascade_output_stagnation_epochs

   Sets the number of cascade output stagnation epochs.
   
   See also:
   		<fann_get_cascade_output_stagnation_epochs>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_output_stagnation_epochs(struct fann *ann, 
															 unsigned int cascade_output_stagnation_epochs);


/* Function: fann_get_cascade_candidate_change_fraction

   The cascade candidate change fraction is a number between 0 and 1 determining how large a fraction
   the <fann_get_MSE> value should change within <fann_get_cascade_candidate_stagnation_epochs> during
   training of the candidate neurons, in order for the training not to stagnate. If the training 
   stagnates, the training of the candidate neurons will be ended and the best candidate will be selected.
   
   This means:
   If the MSE does not change by a fraction of <fann_get_cascade_candidate_change_fraction> during a 
   period of <fann_get_cascade_candidate_stagnation_epochs>, the training of the candidate neurons
   is stopped because the training has stagnated.

   If the cascade candidate change fraction is low, the candidate neurons will be trained more and if the
   fraction is high they will be trained less.
   
   The default cascade candidate change fraction is 0.01, which is equivalent to a 1% change in MSE.

   See also:
   		<fann_set_cascade_candidate_change_fraction>, <fann_get_MSE>, <fann_get_cascade_candidate_stagnation_epochs>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL float FANN_API fann_get_cascade_candidate_change_fraction(struct fann *ann);


/* Function: fann_set_cascade_candidate_change_fraction

   Sets the cascade candidate change fraction.
   
   See also:
   		<fann_get_cascade_candidate_change_fraction>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_candidate_change_fraction(struct fann *ann, 
															 float cascade_candidate_change_fraction);

/* Function: fann_get_cascade_candidate_stagnation_epochs

   The number of cascade candidate stagnation epochs determines the number of epochs training is allowed to
   continue without changing the MSE by a fraction of <fann_get_cascade_candidate_change_fraction>.
   
   See more info about this parameter in <fann_get_cascade_candidate_change_fraction>.

   The default number of cascade candidate stagnation epochs is 12.

   See also:
   		<fann_set_cascade_candidate_stagnation_epochs>, <fann_get_cascade_candidate_change_fraction>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_candidate_stagnation_epochs(struct fann *ann);


/* Function: fann_set_cascade_candidate_stagnation_epochs

   Sets the number of cascade candidate stagnation epochs.
   
   See also:
   		<fann_get_cascade_candidate_stagnation_epochs>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_candidate_stagnation_epochs(struct fann *ann, 
															 unsigned int cascade_candidate_stagnation_epochs);


/* Function: fann_get_cascade_weight_multiplier

   The weight multiplier is a parameter which is used to multiply the weights from the candidate neuron
   before adding the neuron to the neural network. This parameter is usually between 0 and 1, and is used
   to make the training a bit less aggressive.

   The default weight multiplier is 0.4

   See also:
   		<fann_set_cascade_weight_multiplier>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL fann_type FANN_API fann_get_cascade_weight_multiplier(struct fann *ann);


/* Function: fann_set_cascade_weight_multiplier
   
   Sets the weight multiplier.
   
   See also:
   		<fann_get_cascade_weight_multiplier>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_weight_multiplier(struct fann *ann, 
															 fann_type cascade_weight_multiplier);


/* Function: fann_get_cascade_candidate_limit

   The candidate limit is a limit for how much the candidate neuron may be trained.
   The limit is a limit on the proportion between the MSE and candidate score.
   
   Set this to a lower value to avoid overfitting and to a higher if overfitting is
   not a problem.
   
   The default candidate limit is 1000.0

   See also:
   		<fann_set_cascade_candidate_limit>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL fann_type FANN_API fann_get_cascade_candidate_limit(struct fann *ann);


/* Function: fann_set_cascade_candidate_limit

   Sets the candidate limit.
  
   See also:
   		<fann_get_cascade_candidate_limit>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_candidate_limit(struct fann *ann, 
															 fann_type cascade_candidate_limit);


/* Function: fann_get_cascade_max_out_epochs

   The maximum out epochs determines the maximum number of epochs the output connections
   may be trained after adding a new candidate neuron.
   
   The default max out epochs is 150

   See also:
   		<fann_set_cascade_max_out_epochs>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_max_out_epochs(struct fann *ann);


/* Function: fann_set_cascade_max_out_epochs

   Sets the maximum out epochs.

   See also:
   		<fann_get_cascade_max_out_epochs>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_max_out_epochs(struct fann *ann, 
															 unsigned int cascade_max_out_epochs);


/* Function: fann_get_cascade_min_out_epochs

   The minimum out epochs determines the minimum number of epochs the output connections
   must be trained after adding a new candidate neuron.
   
   The default min out epochs is 50

   See also:
   		<fann_set_cascade_min_out_epochs>

	This function appears in FANN >= 2.2.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_min_out_epochs(struct fann *ann);


/* Function: fann_set_cascade_min_out_epochs

   Sets the minimum out epochs.

   See also:
   		<fann_get_cascade_min_out_epochs>

	This function appears in FANN >= 2.2.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_min_out_epochs(struct fann *ann, 
															 unsigned int cascade_min_out_epochs);

/* Function: fann_get_cascade_max_cand_epochs

   The maximum candidate epochs determines the maximum number of epochs the input 
   connections to the candidates may be trained before adding a new candidate neuron.
   
   The default max candidate epochs is 150

   See also:
   		<fann_set_cascade_max_cand_epochs>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_max_cand_epochs(struct fann *ann);


/* Function: fann_set_cascade_max_cand_epochs

   Sets the max candidate epochs.
  
   See also:
   		<fann_get_cascade_max_cand_epochs>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_max_cand_epochs(struct fann *ann, 
															 unsigned int cascade_max_cand_epochs);


/* Function: fann_get_cascade_min_cand_epochs

   The minimum candidate epochs determines the minimum number of epochs the input 
   connections to the candidates may be trained before adding a new candidate neuron.
   
   The default min candidate epochs is 50

   See also:
   		<fann_set_cascade_min_cand_epochs>

	This function appears in FANN >= 2.2.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_min_cand_epochs(struct fann *ann);


/* Function: fann_set_cascade_min_cand_epochs

   Sets the min candidate epochs.
  
   See also:
   		<fann_get_cascade_min_cand_epochs>

	This function appears in FANN >= 2.2.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_min_cand_epochs(struct fann *ann, 
															 unsigned int cascade_min_cand_epochs);

/* Function: fann_get_cascade_num_candidates

   The number of candidates used during training (calculated by multiplying <fann_get_cascade_activation_functions_count>,
   <fann_get_cascade_activation_steepnesses_count> and <fann_get_cascade_num_candidate_groups>). 

   The actual candidates is defined by the <fann_get_cascade_activation_functions> and 
   <fann_get_cascade_activation_steepnesses> arrays. These arrays define the activation functions 
   and activation steepnesses used for the candidate neurons. If there are 2 activation functions
   in the activation function array and 3 steepnesses in the steepness array, then there will be 
   2x3=6 different candidates which will be trained. These 6 different candidates can be copied into
   several candidate groups, where the only difference between these groups is the initial weights.
   If the number of groups is set to 2, then the number of candidate neurons will be 2x3x2=12. The 
   number of candidate groups is defined by <fann_set_cascade_num_candidate_groups>.

   The default number of candidates is 6x4x2 = 48

   See also:
   		<fann_get_cascade_activation_functions>, <fann_get_cascade_activation_functions_count>, 
   		<fann_get_cascade_activation_steepnesses>, <fann_get_cascade_activation_steepnesses_count>,
   		<fann_get_cascade_num_candidate_groups>

	This function appears in FANN >= 2.0.0.
 */ 
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_num_candidates(struct fann *ann);

/* Function: fann_get_cascade_activation_functions_count

   The number of activation functions in the <fann_get_cascade_activation_functions> array.

   The default number of activation functions is 10.

   See also:
   		<fann_get_cascade_activation_functions>, <fann_set_cascade_activation_functions>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_activation_functions_count(struct fann *ann);


/* Function: fann_get_cascade_activation_functions

   The cascade activation functions array is an array of the different activation functions used by
   the candidates. 
   
   See <fann_get_cascade_num_candidates> for a description of which candidate neurons will be 
   generated by this array.
   
   The default activation functions are {FANN_SIGMOID, FANN_SIGMOID_SYMMETRIC, FANN_GAUSSIAN,
   FANN_GAUSSIAN_SYMMETRIC, FANN_ELLIOT, FANN_ELLIOT_SYMMETRIC, FANN_SIN_SYMMETRIC,
   FANN_COS_SYMMETRIC, FANN_SIN, FANN_COS}

   See also:
   		<fann_get_cascade_activation_functions_count>, <fann_set_cascade_activation_functions>,
   		<fann_activationfunc_enum>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL enum fann_activationfunc_enum * FANN_API fann_get_cascade_activation_functions(
															struct fann *ann);


/* Function: fann_set_cascade_activation_functions

   Sets the array of cascade candidate activation functions. The array must be just as long
   as defined by the count.

   See <fann_get_cascade_num_candidates> for a description of which candidate neurons will be 
   generated by this array.

   See also:
   		<fann_get_cascade_activation_steepnesses_count>, <fann_get_cascade_activation_steepnesses>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_activation_functions(struct fann *ann,
														 enum fann_activationfunc_enum *
														 cascade_activation_functions,
														 unsigned int 
														 cascade_activation_functions_count);


/* Function: fann_get_cascade_activation_steepnesses_count

   The number of activation steepnesses in the <fann_get_cascade_activation_functions> array.

   The default number of activation steepnesses is 4.

   See also:
   		<fann_get_cascade_activation_steepnesses>, <fann_set_cascade_activation_functions>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_activation_steepnesses_count(struct fann *ann);


/* Function: fann_get_cascade_activation_steepnesses

   The cascade activation steepnesses array is an array of the different activation functions used by
   the candidates.

   See <fann_get_cascade_num_candidates> for a description of which candidate neurons will be 
   generated by this array.

   The default activation steepnesses is {0.25, 0.50, 0.75, 1.00}

   See also:
   		<fann_set_cascade_activation_steepnesses>, <fann_get_cascade_activation_steepnesses_count>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL fann_type * FANN_API fann_get_cascade_activation_steepnesses(struct fann *ann);
																

/* Function: fann_set_cascade_activation_steepnesses

   Sets the array of cascade candidate activation steepnesses. The array must be just as long
   as defined by the count.

   See <fann_get_cascade_num_candidates> for a description of which candidate neurons will be 
   generated by this array.

   See also:
   		<fann_get_cascade_activation_steepnesses>, <fann_get_cascade_activation_steepnesses_count>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_activation_steepnesses(struct fann *ann,
														   fann_type *
														   cascade_activation_steepnesses,
														   unsigned int 
														   cascade_activation_steepnesses_count);

/* Function: fann_get_cascade_num_candidate_groups

   The number of candidate groups is the number of groups of identical candidates which will be used
   during training.
   
   This number can be used to have more candidates without having to define new parameters for the candidates.
   
   See <fann_get_cascade_num_candidates> for a description of which candidate neurons will be 
   generated by this parameter.
   
   The default number of candidate groups is 2

   See also:
   		<fann_set_cascade_num_candidate_groups>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_num_candidate_groups(struct fann *ann);


/* Function: fann_set_cascade_num_candidate_groups

   Sets the number of candidate groups.

   See also:
   		<fann_get_cascade_num_candidate_groups>

	This function appears in FANN >= 2.0.0.
 */
FANN_EXTERNAL void FANN_API fann_set_cascade_num_candidate_groups(struct fann *ann, 
															 unsigned int cascade_num_candidate_groups);

#endif  /* FIXEDFANN */

#endif
