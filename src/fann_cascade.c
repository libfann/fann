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

#include "config.h"
#include "fann.h"
#include "string.h"

#ifndef FIXEDFANN

/* #define CASCADE_DEBUG */
/* #define CASCADE_DEBUG_FULL */

void fann_print_connections_raw(struct fann *ann)
{
	unsigned int i;

	for(i = 0; i < ann->total_connections_allocated; i++)
	{
		if(i == ann->total_connections)
		{
			printf("* ");
		}
		printf("%f ", ann->weights[i]);
	}
	printf("\n\n");
}

/* Cascade training directly on the training data.
   The connected_neurons pointers are not valid during training,
   but they will be again after training.
 */
FANN_EXTERNAL void FANN_API fann_cascadetrain_on_data(struct fann *ann, struct fann_train_data *data,
										unsigned int max_neurons,
										unsigned int neurons_between_reports,
										float desired_error)
{
	float error;
	unsigned int i;
	unsigned int total_epochs = 0;
	int desired_error_reached;

	if(neurons_between_reports && ann->callback == NULL)
	{
		printf("Max neurons %3d. Desired error: %.6f\n", max_neurons, desired_error);
	}

	for(i = 1; i <= max_neurons; i++)
	{
		/* train output neurons */
		total_epochs += fann_train_outputs(ann, data, desired_error);
		error = fann_get_MSE(ann);
		desired_error_reached = fann_desired_error_reached(ann, desired_error);

		/* print current error */
		if(neurons_between_reports &&
		   (i % neurons_between_reports == 0
			|| i == max_neurons || i == 1 || desired_error_reached == 0))
		{
			if(ann->callback == NULL)
			{
				printf
					("Neurons     %3d. Current error: %.6f. Total error:%8.4f. Epochs %5d. Bit fail %3d",
					 i-1, error, ann->MSE_value, total_epochs, ann->num_bit_fail);
				if((ann->last_layer-2) != ann->first_layer)
				{
					printf(". candidate steepness %.2f. function %s", 
					   (ann->last_layer-2)->first_neuron->activation_steepness,
					   FANN_ACTIVATIONFUNC_NAMES[(ann->last_layer-2)->first_neuron->activation_function]);
				}
				printf("\n");
			}
			else if((*ann->callback) (ann, data, max_neurons, 
				neurons_between_reports, desired_error, total_epochs) == -1) 
			{
				/* you can break the training by returning -1 */
				break;
			}					 
		}

		if(desired_error_reached == 0)
			break;

		if(fann_initialize_candidates(ann) == -1)
		{
			/* Unable to initialize room for candidates */
			break;
		}

		/* train new candidates */
		total_epochs += fann_train_candidates(ann, data);

		/* this installs the best candidate */
		fann_install_candidate(ann);
	}

	/* Train outputs one last time but without any desired error */
	total_epochs += fann_train_outputs(ann, data, 0.0);

	if(neurons_between_reports && ann->callback == NULL)
	{
		printf("Train outputs    Current error: %.6f. Epochs %6d\n", fann_get_MSE(ann),
			   total_epochs);
	}

	/* Set pointers in connected_neurons
	 * This is ONLY done in the end of cascade training,
	 * since there is no need for them during training.
	 */
	fann_set_shortcut_connections(ann);
}

FANN_EXTERNAL void FANN_API fann_cascadetrain_on_file(struct fann *ann, const char *filename,
													  unsigned int max_neurons,
													  unsigned int neurons_between_reports,
													  float desired_error)
{
	struct fann_train_data *data = fann_read_train_from_file(filename);

	if(data == NULL)
	{
		return;
	}
	fann_cascadetrain_on_data(ann, data, max_neurons, neurons_between_reports, desired_error);
	fann_destroy_train(data);
}

int fann_train_outputs(struct fann *ann, struct fann_train_data *data, float desired_error)
{
	float error, initial_error, error_improvement;
	float target_improvement = 0.0;
	float backslide_improvement = -1.0e20f;
	unsigned int i;
	unsigned int max_epochs = ann->cascade_max_out_epochs;
	unsigned int min_epochs = ann->cascade_min_out_epochs;
	unsigned int stagnation = max_epochs;

	/* TODO should perhaps not clear all arrays */
	fann_clear_train_arrays(ann);

	/* run an initial epoch to set the initital error */
	initial_error = fann_train_outputs_epoch(ann, data);

	if(fann_desired_error_reached(ann, desired_error) == 0)
		return 1;

	for(i = 1; i < max_epochs; i++)
	{
		error = fann_train_outputs_epoch(ann, data);

		/*printf("Epoch %6d. Current error: %.6f. Bit fail %d.\n", i, error, ann->num_bit_fail); */

		if(fann_desired_error_reached(ann, desired_error) == 0)
		{
#ifdef CASCADE_DEBUG
			printf("Error %f < %f\n", error, desired_error);
#endif
			return i + 1;
		}

		/* Improvement since start of train */
		error_improvement = initial_error - error;

		/* After any significant change, set a new goal and
		 * allow a new quota of epochs to reach it */
		
		if((target_improvement >= 0 &&
			(error_improvement > target_improvement || error_improvement < backslide_improvement)) ||
		(target_improvement < 0 &&
			(error_improvement < target_improvement || error_improvement > backslide_improvement)))
		{
			/*printf("error_improvement=%f, target_improvement=%f, backslide_improvement=%f, stagnation=%d\n", error_improvement, target_improvement, backslide_improvement, stagnation); */

			target_improvement = error_improvement * (1.0f + ann->cascade_output_change_fraction);
			backslide_improvement = error_improvement * (1.0f - ann->cascade_output_change_fraction);
			stagnation = i + ann->cascade_output_stagnation_epochs;
		}

		/* No improvement in allotted period, so quit */
		if(i >= stagnation && i >= min_epochs)
		{
			return i + 1;
		}
	}

	return max_epochs;
}

float fann_train_outputs_epoch(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;
	
	fann_reset_MSE(ann);

	for(i = 0; i < data->num_data; i++)
	{
		fann_run(ann, data->input[i]);
		fann_compute_MSE(ann, data->output[i]);
		fann_update_slopes_batch(ann, ann->last_layer - 1, ann->last_layer - 1);
	}

	switch (ann->training_algorithm)
	{
		case FANN_TRAIN_RPROP:
			fann_update_weights_irpropm(ann, (ann->last_layer - 1)->first_neuron->first_con,
										ann->total_connections);
			break;
		case FANN_TRAIN_SARPROP:
			fann_update_weights_sarprop(ann, ann->sarprop_epoch, (ann->last_layer - 1)->first_neuron->first_con,
										ann->total_connections);
			++(ann->sarprop_epoch);
			break;
		case FANN_TRAIN_QUICKPROP:
			fann_update_weights_quickprop(ann, data->num_data,
										  (ann->last_layer - 1)->first_neuron->first_con,
										  ann->total_connections);
			break;
		case FANN_TRAIN_BATCH:
		case FANN_TRAIN_INCREMENTAL:
			fann_error((struct fann_error *) ann, FANN_E_CANT_USE_TRAIN_ALG);
	}

	return fann_get_MSE(ann);
}

int fann_reallocate_connections(struct fann *ann, unsigned int total_connections)
{
	/* The connections are allocated, but the pointers inside are
	 * first moved in the end of the cascade training session.
	 */

#ifdef CASCADE_DEBUG
	printf("realloc from %d to %d\n", ann->total_connections_allocated, total_connections);
#endif
	ann->connections =
		(struct fann_neuron **) realloc(ann->connections,
										total_connections * sizeof(struct fann_neuron *));
	if(ann->connections == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->weights = (fann_type *) realloc(ann->weights, total_connections * sizeof(fann_type));
	if(ann->weights == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->train_slopes =
		(fann_type *) realloc(ann->train_slopes, total_connections * sizeof(fann_type));
	if(ann->train_slopes == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->prev_steps = (fann_type *) realloc(ann->prev_steps, total_connections * sizeof(fann_type));
	if(ann->prev_steps == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->prev_train_slopes =
		(fann_type *) realloc(ann->prev_train_slopes, total_connections * sizeof(fann_type));
	if(ann->prev_train_slopes == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	ann->total_connections_allocated = total_connections;

	return 0;
}

int fann_reallocate_neurons(struct fann *ann, unsigned int total_neurons)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neurons;
	unsigned int num_neurons = 0;
	unsigned int num_neurons_so_far = 0;

	neurons =
		(struct fann_neuron *) realloc(ann->first_layer->first_neuron,
									   total_neurons * sizeof(struct fann_neuron));
	ann->total_neurons_allocated = total_neurons;

	if(neurons == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	/* Also allocate room for more train_errors */
	ann->train_errors = (fann_type *) realloc(ann->train_errors, total_neurons * sizeof(fann_type));
	if(ann->train_errors == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return -1;
	}

	if(neurons != ann->first_layer->first_neuron)
	{
		/* Then the memory has moved, also move the pointers */

#ifdef CASCADE_DEBUG_FULL
		printf("Moving neuron pointers\n");
#endif

		/* Move pointers from layers to neurons */
		for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
		{
			num_neurons = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
			layer_it->first_neuron = neurons + num_neurons_so_far;
			layer_it->last_neuron = layer_it->first_neuron + num_neurons;
			num_neurons_so_far += num_neurons;
		}
	}

	return 0;
}

void initialize_candidate_weights(struct fann *ann, unsigned int first_con, unsigned int last_con, float scale_factor)
{
	fann_type prev_step;
	unsigned int i = 0;
	unsigned int bias_weight = (unsigned int)(first_con + (ann->first_layer->last_neuron - ann->first_layer->first_neuron) - 1);

	if(ann->training_algorithm == FANN_TRAIN_RPROP)
		prev_step = ann->rprop_delta_zero;
	else
		prev_step = 0;

	for(i = first_con; i < last_con; i++)
	{
		if(i == bias_weight) 
			ann->weights[i] = fann_rand(-scale_factor, scale_factor);
		else
			ann->weights[i] = fann_rand(0,scale_factor);
					
		ann->train_slopes[i] = 0;
		ann->prev_steps[i] = prev_step;
		ann->prev_train_slopes[i] = 0;
	}
}

int fann_initialize_candidates(struct fann *ann)
{
	/* The candidates are allocated after the normal neurons and connections,
	 * but there is an empty place between the real neurons and the candidate neurons,
	 * so that it will be possible to make room when the chosen candidate are copied in
	 * on the desired place.
	 */
	unsigned int neurons_to_allocate, connections_to_allocate;
	unsigned int num_candidates = fann_get_cascade_num_candidates(ann);
	unsigned int num_neurons = ann->total_neurons + num_candidates + 1;
	unsigned int num_hidden_neurons = ann->total_neurons - ann->num_input - ann->num_output;
	unsigned int candidate_connections_in = ann->total_neurons - ann->num_output;
	unsigned int candidate_connections_out = ann->num_output;

	/* the number of connections going into a and out of a candidate is
	 * ann->total_neurons */
	unsigned int num_connections =
		ann->total_connections + (ann->total_neurons * (num_candidates + 1));
	unsigned int first_candidate_connection = ann->total_connections + ann->total_neurons;
	unsigned int first_candidate_neuron = ann->total_neurons + 1;
	unsigned int connection_it, i, j, k, candidate_index;
	struct fann_neuron *neurons;
	float scale_factor;
	
	/* First make sure that there is enough room, and if not then allocate a
	 * bit more so that we do not need to allocate more room each time.
	 */
	if(num_neurons > ann->total_neurons_allocated)
	{
		/* Then we need to allocate more neurons
		 * Allocate half as many neurons as already exist (at least ten)
		 */
		neurons_to_allocate = num_neurons + num_neurons / 2;
		if(neurons_to_allocate < num_neurons + 10)
		{
			neurons_to_allocate = num_neurons + 10;
		}

		if(fann_reallocate_neurons(ann, neurons_to_allocate) == -1)
		{
			return -1;
		}
	}

	if(num_connections > ann->total_connections_allocated)
	{
		/* Then we need to allocate more connections
		 * Allocate half as many connections as already exist
		 * (at least enough for ten neurons)
		 */
		connections_to_allocate = num_connections + num_connections / 2;
		if(connections_to_allocate < num_connections + ann->total_neurons * 10)
		{
			connections_to_allocate = num_connections + ann->total_neurons * 10;
		}

		if(fann_reallocate_connections(ann, connections_to_allocate) == -1)
		{
			return -1;
		}
	}

	/* Some code to do semi Widrow + Nguyen initialization */
	scale_factor = (float) (2.0 * pow(0.7f * (float)num_hidden_neurons, 1.0f / (float) ann->num_input));
	if(scale_factor > 8)
		scale_factor = 8;
	else if(scale_factor < 0.5)
		scale_factor = 0.5;

	/* Set the neurons.
	 */
	connection_it = first_candidate_connection;
	neurons = ann->first_layer->first_neuron;
	candidate_index = first_candidate_neuron;

	for(i = 0; i < ann->cascade_activation_functions_count; i++)
	{
		for(j = 0; j < ann->cascade_activation_steepnesses_count; j++)
		{
			for(k = 0; k < ann->cascade_num_candidate_groups; k++)
			{
				/* TODO candidates should actually be created both in
				 * the last layer before the output layer, and in a new layer.
				 */
				neurons[candidate_index].value = 0;
				neurons[candidate_index].sum = 0;
				
				neurons[candidate_index].activation_function =
					ann->cascade_activation_functions[i];
				neurons[candidate_index].activation_steepness =
					ann->cascade_activation_steepnesses[j];
				
				neurons[candidate_index].first_con = connection_it;
				connection_it += candidate_connections_in;
				neurons[candidate_index].last_con = connection_it;
				/* We have no specific pointers to the output weights, but they are
				 * available after last_con */
				connection_it += candidate_connections_out;
				ann->train_errors[candidate_index] = 0;
				initialize_candidate_weights(ann, neurons[candidate_index].first_con, neurons[candidate_index].last_con+candidate_connections_out, scale_factor);
				candidate_index++;
			}
		}
	}

	
	/* Now randomize the weights and zero out the arrays that needs zeroing out.
	 */
	 /*
#ifdef CASCADE_DEBUG_FULL
	printf("random cand weight [%d ... %d]\n", first_candidate_connection, num_connections - 1);
#endif

	for(i = first_candidate_connection; i < num_connections; i++)
	{
		
		//ann->weights[i] = fann_random_weight();
		ann->weights[i] = fann_rand(-2.0,2.0);
		ann->train_slopes[i] = 0;
		ann->prev_steps[i] = 0;
		ann->prev_train_slopes[i] = initial_slope;
	}
	*/

	return 0;
}

int fann_train_candidates(struct fann *ann, struct fann_train_data *data)
{
	fann_type best_cand_score = 0.0;
	fann_type target_cand_score = 0.0;
	fann_type backslide_cand_score = -1.0e20f;
	unsigned int i;
	unsigned int max_epochs = ann->cascade_max_cand_epochs;
	unsigned int min_epochs = ann->cascade_min_cand_epochs;
	unsigned int stagnation = max_epochs;

	if(ann->cascade_candidate_scores == NULL)
	{
		ann->cascade_candidate_scores =
			(fann_type *) malloc(fann_get_cascade_num_candidates(ann) * sizeof(fann_type));
		if(ann->cascade_candidate_scores == NULL)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
			return 0;
		}
	}

	for(i = 0; i < max_epochs; i++)
	{
		best_cand_score = fann_train_candidates_epoch(ann, data);

		if(best_cand_score / ann->MSE_value > ann->cascade_candidate_limit)
		{
#ifdef CASCADE_DEBUG
			printf("above candidate limit %f/%f > %f", best_cand_score, ann->MSE_value,
				   ann->cascade_candidate_limit);
#endif
			return i + 1;
		}

		if((best_cand_score > target_cand_score) || (best_cand_score < backslide_cand_score))
		{
#ifdef CASCADE_DEBUG_FULL
			printf("Best candidate score %f, real score: %f\n", ann->MSE_value - best_cand_score,
				   best_cand_score);
			/* printf("best_cand_score=%f, target_cand_score=%f, backslide_cand_score=%f, stagnation=%d\n", best_cand_score, target_cand_score, backslide_cand_score, stagnation); */
#endif

			target_cand_score = best_cand_score * (1.0f + ann->cascade_candidate_change_fraction);
			backslide_cand_score = best_cand_score * (1.0f - ann->cascade_candidate_change_fraction);
			stagnation = i + ann->cascade_candidate_stagnation_epochs;
		}

		/* No improvement in allotted period, so quit */
		if(i >= stagnation && i >= min_epochs)
		{
#ifdef CASCADE_DEBUG
			printf("Stagnation with %d epochs, best candidate score %f, real score: %f\n", i + 1,
				   ann->MSE_value - best_cand_score, best_cand_score);
#endif
			return i + 1;
		}
	}

#ifdef CASCADE_DEBUG
	printf("Max epochs %d reached, best candidate score %f, real score: %f\n", max_epochs,
		   ann->MSE_value - best_cand_score, best_cand_score);
#endif
	return max_epochs;
}

void fann_update_candidate_slopes(struct fann *ann)
{
	struct fann_neuron *neurons = ann->first_layer->first_neuron;
	struct fann_neuron *first_cand = neurons + ann->total_neurons + 1;
	struct fann_neuron *last_cand = first_cand + fann_get_cascade_num_candidates(ann);
	struct fann_neuron *cand_it;
	unsigned int i, j, num_connections;
	unsigned int num_output = ann->num_output;
	fann_type max_sum, cand_sum, activation, derived, error_value, diff, cand_score;
	fann_type *weights, *cand_out_weights, *cand_slopes, *cand_out_slopes;
	fann_type *output_train_errors = ann->train_errors + (ann->total_neurons - ann->num_output);

	for(cand_it = first_cand; cand_it < last_cand; cand_it++)
	{
		cand_score = ann->cascade_candidate_scores[cand_it - first_cand];
		error_value = 0.0;

		/* code more or less stolen from fann_run to fast forward pass
		 */
		cand_sum = 0.0;
		num_connections = cand_it->last_con - cand_it->first_con;
		weights = ann->weights + cand_it->first_con;

		/* unrolled loop start */
		i = num_connections & 3;	/* same as modulo 4 */
		switch (i)
		{
			case 3:
				cand_sum += weights[2] * neurons[2].value;
			case 2:
				cand_sum += weights[1] * neurons[1].value;
			case 1:
				cand_sum += weights[0] * neurons[0].value;
			case 0:
				break;
		}

		for(; i != num_connections; i += 4)
		{
			cand_sum +=
				weights[i] * neurons[i].value +
				weights[i + 1] * neurons[i + 1].value +
				weights[i + 2] * neurons[i + 2].value + weights[i + 3] * neurons[i + 3].value;
		}
		/*
		 * for(i = 0; i < num_connections; i++){
		 * cand_sum += weights[i] * neurons[i].value;
		 * }
		 */
		/* unrolled loop end */

		max_sum = 150/cand_it->activation_steepness;
		if(cand_sum > max_sum)
			cand_sum = max_sum;
		else if(cand_sum < -max_sum)
			cand_sum = -max_sum;
		
		activation =
			fann_activation(ann, cand_it->activation_function, cand_it->activation_steepness,
							cand_sum);
		/* printf("%f = sigmoid(%f);\n", activation, cand_sum); */

		cand_it->sum = cand_sum;
		cand_it->value = activation;

		derived = fann_activation_derived(cand_it->activation_function,
										  cand_it->activation_steepness, activation, cand_sum);

		/* The output weights is located right after the input weights in
		 * the weight array.
		 */
		cand_out_weights = weights + num_connections;

		cand_out_slopes = ann->train_slopes + cand_it->first_con + num_connections;
		for(j = 0; j < num_output; j++)
		{
			diff = (activation * cand_out_weights[j]) - output_train_errors[j];
#ifdef CASCADE_DEBUG_FULL
			/* printf("diff = %f = (%f * %f) - %f;\n", diff, activation, cand_out_weights[j], output_train_errors[j]); */
#endif
			cand_out_slopes[j] -= 2.0f * diff * activation;
#ifdef CASCADE_DEBUG_FULL
			/* printf("cand_out_slopes[%d] <= %f += %f * %f;\n", j, cand_out_slopes[j], diff, activation); */
#endif
			error_value += diff * cand_out_weights[j];
			cand_score -= (diff * diff);
#ifdef CASCADE_DEBUG_FULL
			/* printf("cand_score[%d][%d] = %f -= (%f * %f)\n", cand_it - first_cand, j, cand_score, diff, diff); */

			printf("cand[%d]: error=%f, activation=%f, diff=%f, slope=%f\n", cand_it - first_cand,
				   output_train_errors[j], (activation * cand_out_weights[j]), diff,
				   -2.0 * diff * activation);
#endif
		}

		ann->cascade_candidate_scores[cand_it - first_cand] = cand_score;
		error_value *= derived;

		cand_slopes = ann->train_slopes + cand_it->first_con;
		for(i = 0; i < num_connections; i++)
		{
			cand_slopes[i] -= error_value * neurons[i].value;
		}
	}
}

void fann_update_candidate_weights(struct fann *ann, unsigned int num_data)
{
	struct fann_neuron *first_cand = (ann->last_layer - 1)->last_neuron + 1;	/* there is an empty neuron between the actual neurons and the candidate neuron */
	struct fann_neuron *last_cand = first_cand + fann_get_cascade_num_candidates(ann) - 1;

	switch (ann->training_algorithm)
	{
		case FANN_TRAIN_RPROP:
			fann_update_weights_irpropm(ann, first_cand->first_con,
										last_cand->last_con + ann->num_output);
			break;
		case FANN_TRAIN_SARPROP:
			/* TODO: increase epoch? */
			fann_update_weights_sarprop(ann, ann->sarprop_epoch, first_cand->first_con,
										last_cand->last_con + ann->num_output);
			break;
		case FANN_TRAIN_QUICKPROP:
			fann_update_weights_quickprop(ann, num_data, first_cand->first_con,
										  last_cand->last_con + ann->num_output);
			break;
		case FANN_TRAIN_BATCH:
		case FANN_TRAIN_INCREMENTAL:
			fann_error((struct fann_error *) ann, FANN_E_CANT_USE_TRAIN_ALG);
			break;
	}
}

fann_type fann_train_candidates_epoch(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i, j;
	unsigned int best_candidate;
	fann_type best_score;
	unsigned int num_cand = fann_get_cascade_num_candidates(ann);
	fann_type *output_train_errors = ann->train_errors + (ann->total_neurons - ann->num_output);
	struct fann_neuron *output_neurons = (ann->last_layer - 1)->first_neuron;

	for(i = 0; i < num_cand; i++)
	{
		/* The ann->MSE_value is actually the sum squared error */
		ann->cascade_candidate_scores[i] = ann->MSE_value;
	}
	/*printf("start score: %f\n", ann->MSE_value); */

	for(i = 0; i < data->num_data; i++)
	{
		fann_run(ann, data->input[i]);

		for(j = 0; j < ann->num_output; j++)
		{
			/* TODO only debug, but the error is in opposite direction, this might be usefull info */
			/*          if(output_train_errors[j] != (ann->output[j] - data->output[i][j])){
			 * printf("difference in calculated error at %f != %f; %f = %f - %f;\n", output_train_errors[j], (ann->output[j] - data->output[i][j]), output_train_errors[j], ann->output[j], data->output[i][j]);
			 * } */

			/*
			 * output_train_errors[j] = (data->output[i][j] - ann->output[j])/2;
			 * output_train_errors[j] = ann->output[j] - data->output[i][j];
			 */

			output_train_errors[j] = (data->output[i][j] - ann->output[j]);

			switch (output_neurons[j].activation_function)
			{
				case FANN_LINEAR_PIECE_SYMMETRIC:
				case FANN_SIGMOID_SYMMETRIC:
				case FANN_SIGMOID_SYMMETRIC_STEPWISE:
				case FANN_THRESHOLD_SYMMETRIC:
				case FANN_ELLIOT_SYMMETRIC:
				case FANN_GAUSSIAN_SYMMETRIC:
				case FANN_SIN_SYMMETRIC:
				case FANN_COS_SYMMETRIC:
					output_train_errors[j] /= 2.0;
					break;
				case FANN_LINEAR:
				case FANN_THRESHOLD:
				case FANN_SIGMOID:
				case FANN_SIGMOID_STEPWISE:
				case FANN_GAUSSIAN:
				case FANN_GAUSSIAN_STEPWISE:
				case FANN_ELLIOT:
				case FANN_LINEAR_PIECE:
				case FANN_SIN:
				case FANN_COS:
					break;
			}
		}

		fann_update_candidate_slopes(ann);
	}

	fann_update_candidate_weights(ann, data->num_data);

	/* find the best candidate score */
	best_candidate = 0;
	best_score = ann->cascade_candidate_scores[best_candidate];
	for(i = 1; i < num_cand; i++)
	{
		/*struct fann_neuron *cand = ann->first_layer->first_neuron + ann->total_neurons + 1 + i;
		 * printf("candidate[%d] = activation: %s, steepness: %f, score: %f\n", 
		 * i, FANN_ACTIVATIONFUNC_NAMES[cand->activation_function], 
		 * cand->activation_steepness, ann->cascade_candidate_scores[i]); */

		if(ann->cascade_candidate_scores[i] > best_score)
		{
			best_candidate = i;
			best_score = ann->cascade_candidate_scores[best_candidate];
		}
	}

	ann->cascade_best_candidate = ann->total_neurons + best_candidate + 1;
#ifdef CASCADE_DEBUG
	printf("Best candidate[%d]: with score %f, real score: %f\n", best_candidate,
		   ann->MSE_value - best_score, best_score);
#endif

	return best_score;
}

/* add a layer at the position pointed to by *layer */
struct fann_layer *fann_add_layer(struct fann *ann, struct fann_layer *layer)
{
	int layer_pos = (int)(layer - ann->first_layer);
	int num_layers = (int)(ann->last_layer - ann->first_layer + 1);
	int i;

	/* allocate the layer */
	struct fann_layer *layers =
		(struct fann_layer *) realloc(ann->first_layer, num_layers * sizeof(struct fann_layer));
	if(layers == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	/* copy layers so that the free space is at the right location */
	for(i = num_layers - 1; i >= layer_pos; i--)
	{
		layers[i] = layers[i - 1];
	}

	/* the newly allocated layer is empty */
	layers[layer_pos].first_neuron = layers[layer_pos + 1].first_neuron;
	layers[layer_pos].last_neuron = layers[layer_pos + 1].first_neuron;

	/* Set the ann pointers correctly */
	ann->first_layer = layers;
	ann->last_layer = layers + num_layers;

#ifdef CASCADE_DEBUG_FULL
	printf("add layer at pos %d\n", layer_pos);
#endif

	return layers + layer_pos;
}

void fann_set_shortcut_connections(struct fann *ann)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, **neuron_pointers, *neurons;
	unsigned int num_connections = 0, i;

	neuron_pointers = ann->connections;
	neurons = ann->first_layer->first_neuron;

	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
	{
		for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++)
		{

			neuron_pointers += num_connections;
			num_connections = neuron_it->last_con - neuron_it->first_con;

			for(i = 0; i != num_connections; i++)
			{
				neuron_pointers[i] = neurons + i;
			}
		}
	}
}

void fann_add_candidate_neuron(struct fann *ann, struct fann_layer *layer)
{
	unsigned int num_connections_in = (unsigned int)(layer->first_neuron - ann->first_layer->first_neuron);
	unsigned int num_connections_out = (unsigned int)((ann->last_layer - 1)->last_neuron - (layer + 1)->first_neuron);
	unsigned int num_connections_move = num_connections_out + num_connections_in;

	unsigned int candidate_con, candidate_output_weight;
	int i;

	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, *neuron_place, *candidate;

	/* We know that there is enough room for the new neuron
	 * (the candidates are in the same arrays), so move
	 * the last neurons to make room for this neuron.
	 */

	/* first move the pointers to neurons in the layer structs */
	for(layer_it = ann->last_layer - 1; layer_it != layer; layer_it--)
	{
#ifdef CASCADE_DEBUG_FULL
		printf("move neuron pointers in layer %d, first(%d -> %d), last(%d -> %d)\n",
			   layer_it - ann->first_layer,
			   layer_it->first_neuron - ann->first_layer->first_neuron,
			   layer_it->first_neuron - ann->first_layer->first_neuron + 1,
			   layer_it->last_neuron - ann->first_layer->first_neuron,
			   layer_it->last_neuron - ann->first_layer->first_neuron + 1);
#endif
		layer_it->first_neuron++;
		layer_it->last_neuron++;
	}

	/* also move the last neuron in the layer that needs the neuron added */
	layer->last_neuron++;

	/* this is the place that should hold the new neuron */
	neuron_place = layer->last_neuron - 1;

#ifdef CASCADE_DEBUG_FULL
	printf("num_connections_in=%d, num_connections_out=%d\n", num_connections_in,
		   num_connections_out);
#endif

	candidate = ann->first_layer->first_neuron + ann->cascade_best_candidate;

	/* the output weights for the candidates are located after the input weights */
	candidate_output_weight = candidate->last_con;

	/* move the actual output neurons and the indexes to the connection arrays */
	for(neuron_it = (ann->last_layer - 1)->last_neuron - 1; neuron_it != neuron_place; neuron_it--)
	{
#ifdef CASCADE_DEBUG_FULL
		printf("move neuron %d -> %d\n", neuron_it - ann->first_layer->first_neuron - 1,
			   neuron_it - ann->first_layer->first_neuron);
#endif
		*neuron_it = *(neuron_it - 1);

		/* move the weights */
#ifdef CASCADE_DEBUG_FULL
		printf("move weight[%d ... %d] -> weight[%d ... %d]\n", neuron_it->first_con,
			   neuron_it->last_con - 1, neuron_it->first_con + num_connections_move - 1,
			   neuron_it->last_con + num_connections_move - 2);
#endif
		for(i = neuron_it->last_con - 1; i >= (int)neuron_it->first_con; i--)
		{
#ifdef CASCADE_DEBUG_FULL
			printf("move weight[%d] = weight[%d]\n", i + num_connections_move - 1, i);
#endif
			ann->weights[i + num_connections_move - 1] = ann->weights[i];
		}

		/* move the indexes to weights */
		neuron_it->last_con += num_connections_move;
		num_connections_move--;
		neuron_it->first_con += num_connections_move;

		/* set the new weight to the newly allocated neuron */
		ann->weights[neuron_it->last_con - 1] =
			(ann->weights[candidate_output_weight]) * ann->cascade_weight_multiplier;
		candidate_output_weight++;
	}

	/* Now inititalize the actual neuron */
	neuron_place->value = 0;
	neuron_place->sum = 0;
	neuron_place->activation_function = candidate->activation_function;
	neuron_place->activation_steepness = candidate->activation_steepness;
	neuron_place->last_con = (neuron_place + 1)->first_con;
	neuron_place->first_con = neuron_place->last_con - num_connections_in;
#ifdef CASCADE_DEBUG_FULL
	printf("neuron[%d] = weights[%d ... %d] activation: %s, steepness: %f\n",
		   neuron_place - ann->first_layer->first_neuron, neuron_place->first_con,
		   neuron_place->last_con - 1, FANN_ACTIVATIONFUNC_NAMES[neuron_place->activation_function],
		   neuron_place->activation_steepness);/* TODO remove */
#endif

	candidate_con = candidate->first_con;
	/* initialize the input weights at random */
#ifdef CASCADE_DEBUG_FULL
	printf("move cand weights[%d ... %d] -> [%d ... %d]\n", candidate_con,
		   candidate_con + num_connections_in - 1, neuron_place->first_con,
		   neuron_place->last_con - 1);
#endif

	for(i = 0; i < (int)num_connections_in; i++)
	{
		ann->weights[i + neuron_place->first_con] = ann->weights[i + candidate_con];
#ifdef CASCADE_DEBUG_FULL
		printf("move weights[%d] -> weights[%d] (%f)\n", i + candidate_con,
			   i + neuron_place->first_con, ann->weights[i + neuron_place->first_con]);
#endif
	}

	/* Change some of main variables */
	ann->total_neurons++;
	ann->total_connections += num_connections_in + num_connections_out;

	return;
}

void fann_install_candidate(struct fann *ann)
{
	struct fann_layer *layer;

	layer = fann_add_layer(ann, ann->last_layer - 1);
	fann_add_candidate_neuron(ann, layer);
	return;
}

#endif /* FIXEDFANN */

FANN_EXTERNAL unsigned int FANN_API fann_get_cascade_num_candidates(struct fann *ann)
{
	return ann->cascade_activation_functions_count *
		ann->cascade_activation_steepnesses_count *
		ann->cascade_num_candidate_groups;
}

FANN_GET_SET(float, cascade_output_change_fraction)
FANN_GET_SET(unsigned int, cascade_output_stagnation_epochs)
FANN_GET_SET(float, cascade_candidate_change_fraction)
FANN_GET_SET(unsigned int, cascade_candidate_stagnation_epochs)
FANN_GET_SET(unsigned int, cascade_num_candidate_groups)
FANN_GET_SET(fann_type, cascade_weight_multiplier)
FANN_GET_SET(fann_type, cascade_candidate_limit)
FANN_GET_SET(unsigned int, cascade_max_out_epochs)
FANN_GET_SET(unsigned int, cascade_max_cand_epochs)
FANN_GET_SET(unsigned int, cascade_min_out_epochs)
FANN_GET_SET(unsigned int, cascade_min_cand_epochs)

FANN_GET(unsigned int, cascade_activation_functions_count)
FANN_GET(enum fann_activationfunc_enum *, cascade_activation_functions)

FANN_EXTERNAL void FANN_API fann_set_cascade_activation_functions(struct fann *ann,
														 enum fann_activationfunc_enum *
														 cascade_activation_functions,
														 unsigned int 
														 cascade_activation_functions_count)
{
	if(ann->cascade_activation_functions_count != cascade_activation_functions_count)
	{
		ann->cascade_activation_functions_count = cascade_activation_functions_count;
		
		/* reallocate mem */
		ann->cascade_activation_functions = 
			(enum fann_activationfunc_enum *)realloc(ann->cascade_activation_functions, 
			ann->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
		if(ann->cascade_activation_functions == NULL)
		{
			fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	}
	
	memmove(ann->cascade_activation_functions, cascade_activation_functions, 
		ann->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
}

FANN_GET(unsigned int, cascade_activation_steepnesses_count)
FANN_GET(fann_type *, cascade_activation_steepnesses)

FANN_EXTERNAL void FANN_API fann_set_cascade_activation_steepnesses(struct fann *ann,
														   fann_type *
														   cascade_activation_steepnesses,
														   unsigned int 
														   cascade_activation_steepnesses_count)
{
	if(ann->cascade_activation_steepnesses_count != cascade_activation_steepnesses_count)
	{
		ann->cascade_activation_steepnesses_count = cascade_activation_steepnesses_count;
		
		/* reallocate mem */
		ann->cascade_activation_steepnesses = 
			(fann_type *)realloc(ann->cascade_activation_steepnesses, 
			ann->cascade_activation_steepnesses_count * sizeof(fann_type));
		if(ann->cascade_activation_steepnesses == NULL)
		{
			fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
			return;
		}
	}
	
	memmove(ann->cascade_activation_steepnesses, cascade_activation_steepnesses, 
		ann->cascade_activation_steepnesses_count * sizeof(fann_type));
}
