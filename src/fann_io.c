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
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "config.h"
#include "fann.h"
#include "fann_data.h"

/* Create a network from a configuration file.
 */
FANN_EXTERNAL struct fann *FANN_API fann_create_from_file(const char *configuration_file)
{
	struct fann *ann;
	FILE *conf = fopen(configuration_file, "r");

	if(!conf)
	{
		fann_error(NULL, FANN_E_CANT_OPEN_CONFIG_R, configuration_file);
		return NULL;
	}
	ann = fann_create_from_fd(conf, configuration_file);
	fclose(conf);
	return ann;
}

/* Save the network.
 */
FANN_EXTERNAL int FANN_API fann_save(struct fann *ann, const char *configuration_file)
{
	return fann_save_internal(ann, configuration_file, 0);
}

/* Save the network as fixed point data.
 */
FANN_EXTERNAL int FANN_API fann_save_to_fixed(struct fann *ann, const char *configuration_file)
{
	return fann_save_internal(ann, configuration_file, 1);
}

/* INTERNAL FUNCTION
   Used to save the network to a file.
 */
int fann_save_internal(struct fann *ann, const char *configuration_file, unsigned int save_as_fixed)
{
	int retval;
	FILE *conf = fopen(configuration_file, "w+");

	if(!conf)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_OPEN_CONFIG_W, configuration_file);
		return -1;
	}
	retval = fann_save_internal_fd(ann, conf, configuration_file, save_as_fixed);
	fclose(conf);
	return retval;
}

/* INTERNAL FUNCTION
   Used to save the network to a file descriptor.
 */
int fann_save_internal_fd(struct fann *ann, FILE * conf, const char *configuration_file,
						  unsigned int save_as_fixed)
{
	struct fann_layer *layer_it;
	int calculated_decimal_point = 0;
	struct fann_neuron *neuron_it, *first_neuron;
	fann_type *weights;
	struct fann_neuron **connected_neurons;
	unsigned int i = 0;

#ifndef FIXEDFANN
	/* variabels for use when saving floats as fixed point variabels */
	unsigned int decimal_point = 0;
	unsigned int fixed_multiplier = 0;
	fann_type max_possible_value = 0;
	unsigned int bits_used_for_max = 0;
	fann_type current_max_value = 0;
#endif

#ifndef FIXEDFANN
	if(save_as_fixed)
	{
		/* save the version information */
		fprintf(conf, FANN_FIX_VERSION "\n");
	}
	else
	{
		/* save the version information */
		fprintf(conf, FANN_FLO_VERSION "\n");
	}
#else
	/* save the version information */
	fprintf(conf, FANN_FIX_VERSION "\n");
#endif

#ifndef FIXEDFANN
	if(save_as_fixed)
	{
		/* calculate the maximal possible shift value */

		for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
		{
			for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++)
			{
				/* look at all connections to each neurons, and see how high a value we can get */
				current_max_value = 0;
				for(i = neuron_it->first_con; i != neuron_it->last_con; i++)
				{
					current_max_value += fann_abs(ann->weights[i]);
				}

				if(current_max_value > max_possible_value)
				{
					max_possible_value = current_max_value;
				}
			}
		}

		for(bits_used_for_max = 0; max_possible_value >= 1; bits_used_for_max++)
		{
			max_possible_value /= 2.0;
		}

		/* The maximum number of bits we shift the fix point, is the number
		 * of bits in a integer, minus one for the sign, one for the minus
		 * in stepwise, and minus the bits used for the maximum.
		 * This is devided by two, to allow multiplication of two fixed
		 * point numbers.
		 */
		calculated_decimal_point = (sizeof(int) * 8 - 2 - bits_used_for_max) / 2;

		if(calculated_decimal_point < 0)
		{
			decimal_point = 0;
		}
		else
		{
			decimal_point = calculated_decimal_point;
		}

		fixed_multiplier = 1 << decimal_point;

#ifdef DEBUG
		printf("calculated_decimal_point=%d, decimal_point=%u, bits_used_for_max=%u\n",
			   calculated_decimal_point, decimal_point, bits_used_for_max);
#endif

		/* save the decimal_point on a seperate line */
		fprintf(conf, "decimal_point=%u\n", decimal_point);
	}
#else
	/* save the decimal_point on a seperate line */
	fprintf(conf, "decimal_point=%u\n", ann->decimal_point);

#endif

	/* Save network parameters */
	fprintf(conf, "num_layers=%d\n", (int)(ann->last_layer - ann->first_layer));
	fprintf(conf, "learning_rate=%f\n", ann->learning_rate);
	fprintf(conf, "connection_rate=%f\n", ann->connection_rate);
	fprintf(conf, "network_type=%u\n", ann->network_type);
	
	fprintf(conf, "learning_momentum=%f\n", ann->learning_momentum);
	fprintf(conf, "training_algorithm=%u\n", ann->training_algorithm);
	fprintf(conf, "train_error_function=%u\n", ann->train_error_function);
	fprintf(conf, "train_stop_function=%u\n", ann->train_stop_function);
	fprintf(conf, "cascade_output_change_fraction=%f\n", ann->cascade_output_change_fraction);
	fprintf(conf, "quickprop_decay=%f\n", ann->quickprop_decay);
	fprintf(conf, "quickprop_mu=%f\n", ann->quickprop_mu);
	fprintf(conf, "rprop_increase_factor=%f\n", ann->rprop_increase_factor);
	fprintf(conf, "rprop_decrease_factor=%f\n", ann->rprop_decrease_factor);
	fprintf(conf, "rprop_delta_min=%f\n", ann->rprop_delta_min);
	fprintf(conf, "rprop_delta_max=%f\n", ann->rprop_delta_max);
	fprintf(conf, "rprop_delta_zero=%f\n", ann->rprop_delta_zero);
	fprintf(conf, "cascade_output_stagnation_epochs=%u\n", ann->cascade_output_stagnation_epochs);
	fprintf(conf, "cascade_candidate_change_fraction=%f\n", ann->cascade_candidate_change_fraction);
	fprintf(conf, "cascade_candidate_stagnation_epochs=%u\n", ann->cascade_candidate_stagnation_epochs);
	fprintf(conf, "cascade_max_out_epochs=%u\n", ann->cascade_max_out_epochs);
	fprintf(conf, "cascade_min_out_epochs=%u\n", ann->cascade_min_out_epochs);
	fprintf(conf, "cascade_max_cand_epochs=%u\n", ann->cascade_max_cand_epochs);	
	fprintf(conf, "cascade_min_cand_epochs=%u\n", ann->cascade_min_cand_epochs);	
	fprintf(conf, "cascade_num_candidate_groups=%u\n", ann->cascade_num_candidate_groups);

#ifndef FIXEDFANN
	if(save_as_fixed)
	{
		fprintf(conf, "bit_fail_limit=%u\n", (int) floor((ann->bit_fail_limit * fixed_multiplier) + 0.5));
		fprintf(conf, "cascade_candidate_limit=%u\n", (int) floor((ann->cascade_candidate_limit * fixed_multiplier) + 0.5));
		fprintf(conf, "cascade_weight_multiplier=%u\n", (int) floor((ann->cascade_weight_multiplier * fixed_multiplier) + 0.5));
	}
	else
#endif	
	{
		fprintf(conf, "bit_fail_limit="FANNPRINTF"\n", ann->bit_fail_limit);
		fprintf(conf, "cascade_candidate_limit="FANNPRINTF"\n", ann->cascade_candidate_limit);
		fprintf(conf, "cascade_weight_multiplier="FANNPRINTF"\n", ann->cascade_weight_multiplier);
	}

	fprintf(conf, "cascade_activation_functions_count=%u\n", ann->cascade_activation_functions_count);
	fprintf(conf, "cascade_activation_functions=");
	for(i = 0; i < ann->cascade_activation_functions_count; i++)
		fprintf(conf, "%u ", ann->cascade_activation_functions[i]);
	fprintf(conf, "\n");
	
	fprintf(conf, "cascade_activation_steepnesses_count=%u\n", ann->cascade_activation_steepnesses_count);
	fprintf(conf, "cascade_activation_steepnesses=");
	for(i = 0; i < ann->cascade_activation_steepnesses_count; i++)
	{
#ifndef FIXEDFANN
		if(save_as_fixed)
			fprintf(conf, "%u ", (int) floor((ann->cascade_activation_steepnesses[i] * fixed_multiplier) + 0.5));
		else
#endif	
			fprintf(conf, FANNPRINTF" ", ann->cascade_activation_steepnesses[i]);
	}
	fprintf(conf, "\n");

	fprintf(conf, "layer_sizes=");
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		/* the number of neurons in the layers (in the last layer, there is always one too many neurons, because of an unused bias) */
		fprintf(conf, "%d ", (int)(layer_it->last_neuron - layer_it->first_neuron));
	}
	fprintf(conf, "\n");

#ifndef FIXEDFANN
	/* 2.1 */
	#define SCALE_SAVE( what, where )										\
		fprintf( conf, #what "_" #where "=" );								\
		for( i = 0; i < ann->num_##where##put; i++ )						\
			fprintf( conf, "%f ", ann->what##_##where[ i ] );				\
		fprintf( conf, "\n" );

	if(!save_as_fixed)
	{
		if(ann->scale_mean_in != NULL)
		{
			fprintf(conf, "scale_included=1\n");
			SCALE_SAVE( scale_mean,			in )
			SCALE_SAVE( scale_deviation,	in )
			SCALE_SAVE( scale_new_min,		in )
			SCALE_SAVE( scale_factor,		in )
		
			SCALE_SAVE( scale_mean,			out )
			SCALE_SAVE( scale_deviation,	out )
			SCALE_SAVE( scale_new_min,		out )
			SCALE_SAVE( scale_factor,		out )
		}
		else
			fprintf(conf, "scale_included=0\n");
	}
#undef SCALE_SAVE
#endif	

	/* 2.0 */
	fprintf(conf, "neurons (num_inputs, activation_function, activation_steepness)=");
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		/* the neurons */
		for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++)
		{
#ifndef FIXEDFANN
			if(save_as_fixed)
			{
				fprintf(conf, "(%u, %u, %u) ", neuron_it->last_con - neuron_it->first_con,
						neuron_it->activation_function,
						(int) floor((neuron_it->activation_steepness * fixed_multiplier) + 0.5));
			}
			else
			{
				fprintf(conf, "(%u, %u, " FANNPRINTF ") ", neuron_it->last_con - neuron_it->first_con,
						neuron_it->activation_function, neuron_it->activation_steepness);
			}
#else
			fprintf(conf, "(%u, %u, " FANNPRINTF ") ", neuron_it->last_con - neuron_it->first_con,
					neuron_it->activation_function, neuron_it->activation_steepness);
#endif
		}
	}
	fprintf(conf, "\n");

	connected_neurons = ann->connections;
	weights = ann->weights;
	first_neuron = ann->first_layer->first_neuron;

	/* Now save all the connections.
	 * We only need to save the source and the weight,
	 * since the destination is given by the order.
	 * 
	 * The weight is not saved binary due to differences
	 * in binary definition of floating point numbers.
	 * Especially an iPAQ does not use the same binary
	 * representation as an i386 machine.
	 */
	fprintf(conf, "connections (connected_to_neuron, weight)=");
	for(i = 0; i < ann->total_connections; i++)
	{
#ifndef FIXEDFANN
		if(save_as_fixed)
		{
			/* save the connection "(source weight) " */
			fprintf(conf, "(%d, %d) ",
					(int)(connected_neurons[i] - first_neuron),
					(int) floor((weights[i] * fixed_multiplier) + 0.5));
		}
		else
		{
			/* save the connection "(source weight) " */
			fprintf(conf, "(%d, " FANNPRINTF ") ", (int)(connected_neurons[i] - first_neuron), weights[i]);
		}
#else
		/* save the connection "(source weight) " */
		fprintf(conf, "(%d, " FANNPRINTF ") ", (int)(connected_neurons[i] - first_neuron), weights[i]);
#endif

	}
	fprintf(conf, "\n");

	return calculated_decimal_point;
}

struct fann *fann_create_from_fd_1_1(FILE * conf, const char *configuration_file);

#define fann_scanf(type, name, val) \
{ \
	if(fscanf(conf, name"="type"\n", val) != 1) \
	{ \
		fann_error(NULL, FANN_E_CANT_READ_CONFIG, name, configuration_file); \
		fann_destroy(ann); \
		return NULL; \
	} \
}

#define fann_skip(name) \
{ \
	if(fscanf(conf, name) != 0) \
	{ \
		fann_error(NULL, FANN_E_CANT_READ_CONFIG, name, configuration_file); \
		fann_destroy(ann); \
		return NULL; \
	} \
}

/* INTERNAL FUNCTION
   Create a network from a configuration file descriptor.
 */
struct fann *fann_create_from_fd(FILE * conf, const char *configuration_file)
{
	unsigned int num_layers, layer_size, input_neuron, i, num_connections;
	unsigned int tmpVal;
#ifdef FIXEDFANN
	unsigned int decimal_point, multiplier;
#else
	unsigned int scale_included;
#endif
	struct fann_neuron *first_neuron, *neuron_it, *last_neuron, **connected_neurons;
	fann_type *weights;
	struct fann_layer *layer_it;
	struct fann *ann = NULL;

	char *read_version;

	read_version = (char *) calloc(strlen(FANN_CONF_VERSION "\n"), 1);
	if(read_version == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	if(fread(read_version, 1, strlen(FANN_CONF_VERSION "\n"), conf) == 1)
	{
	        fann_error(NULL, FANN_E_CANT_READ_CONFIG, "FANN_VERSION", configuration_file);
		return NULL;
	}

	/* compares the version information */
	if(strncmp(read_version, FANN_CONF_VERSION "\n", strlen(FANN_CONF_VERSION "\n")) != 0)
	{
#ifdef FIXEDFANN
		if(strncmp(read_version, "FANN_FIX_1.1\n", strlen("FANN_FIX_1.1\n")) == 0)
		{
#else
		if(strncmp(read_version, "FANN_FLO_1.1\n", strlen("FANN_FLO_1.1\n")) == 0)
		{
#endif
			free(read_version);
			return fann_create_from_fd_1_1(conf, configuration_file);
		}

#ifndef FIXEDFANN
		/* Maintain compatibility with 2.0 version that doesnt have scale parameters. */
		if(strncmp(read_version, "FANN_FLO_2.0\n", strlen("FANN_FLO_2.0\n")) != 0 &&
		   strncmp(read_version, "FANN_FLO_2.1\n", strlen("FANN_FLO_2.1\n")) != 0)
#else
		if(strncmp(read_version, "FANN_FIX_2.0\n", strlen("FANN_FIX_2.0\n")) != 0 &&
		   strncmp(read_version, "FANN_FIX_2.1\n", strlen("FANN_FIX_2.1\n")) != 0)
#endif
		{
			free(read_version);
			fann_error(NULL, FANN_E_WRONG_CONFIG_VERSION, configuration_file);

			return NULL;
		}
	}

	free(read_version);

#ifdef FIXEDFANN
	fann_scanf("%u", "decimal_point", &decimal_point);
	multiplier = 1 << decimal_point;
#endif

	fann_scanf("%u", "num_layers", &num_layers);

	ann = fann_allocate_structure(num_layers);
	if(ann == NULL)
	{
		return NULL;
	}

	fann_scanf("%f", "learning_rate", &ann->learning_rate);
	fann_scanf("%f", "connection_rate", &ann->connection_rate);
	fann_scanf("%u", "network_type", &tmpVal);
	ann->network_type = (enum fann_nettype_enum)tmpVal;
	fann_scanf("%f", "learning_momentum", &ann->learning_momentum);
	fann_scanf("%u", "training_algorithm", &tmpVal);
	ann->training_algorithm = (enum fann_train_enum)tmpVal;
	fann_scanf("%u", "train_error_function", &tmpVal);
	ann->train_error_function = (enum fann_errorfunc_enum)tmpVal;
	fann_scanf("%u", "train_stop_function", &tmpVal);
	ann->train_stop_function = (enum fann_stopfunc_enum)tmpVal;
	fann_scanf("%f", "cascade_output_change_fraction", &ann->cascade_output_change_fraction);
	fann_scanf("%f", "quickprop_decay", &ann->quickprop_decay);
	fann_scanf("%f", "quickprop_mu", &ann->quickprop_mu);
	fann_scanf("%f", "rprop_increase_factor", &ann->rprop_increase_factor);
	fann_scanf("%f", "rprop_decrease_factor", &ann->rprop_decrease_factor);
	fann_scanf("%f", "rprop_delta_min", &ann->rprop_delta_min);
	fann_scanf("%f", "rprop_delta_max", &ann->rprop_delta_max);
	fann_scanf("%f", "rprop_delta_zero", &ann->rprop_delta_zero);
	fann_scanf("%u", "cascade_output_stagnation_epochs", &ann->cascade_output_stagnation_epochs);
	fann_scanf("%f", "cascade_candidate_change_fraction", &ann->cascade_candidate_change_fraction);
	fann_scanf("%u", "cascade_candidate_stagnation_epochs", &ann->cascade_candidate_stagnation_epochs);
	fann_scanf("%u", "cascade_max_out_epochs", &ann->cascade_max_out_epochs);
	fann_scanf("%u", "cascade_min_out_epochs", &ann->cascade_min_out_epochs);
	fann_scanf("%u", "cascade_max_cand_epochs", &ann->cascade_max_cand_epochs);	
	fann_scanf("%u", "cascade_min_cand_epochs", &ann->cascade_min_cand_epochs);	
	fann_scanf("%u", "cascade_num_candidate_groups", &ann->cascade_num_candidate_groups);

	fann_scanf(FANNSCANF, "bit_fail_limit", &ann->bit_fail_limit);
	fann_scanf(FANNSCANF, "cascade_candidate_limit", &ann->cascade_candidate_limit);
	fann_scanf(FANNSCANF, "cascade_weight_multiplier", &ann->cascade_weight_multiplier);


	fann_scanf("%u", "cascade_activation_functions_count", &ann->cascade_activation_functions_count);

	/* reallocate mem */
	ann->cascade_activation_functions = 
		(enum fann_activationfunc_enum *)realloc(ann->cascade_activation_functions, 
		ann->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
	if(ann->cascade_activation_functions == NULL)
	{
		fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy(ann);
		return NULL;
	}


	fann_skip("cascade_activation_functions=");
	for(i = 0; i < ann->cascade_activation_functions_count; i++)
	{
		if(fscanf(conf, "%u ", (unsigned int *)&ann->cascade_activation_functions[i]) != 1)
		{
			fann_error(NULL, FANN_E_CANT_READ_CONFIG, "cascade_activation_functions", configuration_file);
			fann_destroy(ann);
			return NULL;
		}
	}

	fann_scanf("%u", "cascade_activation_steepnesses_count", &ann->cascade_activation_steepnesses_count);

	/* reallocate mem */
	ann->cascade_activation_steepnesses = 
		(fann_type *)realloc(ann->cascade_activation_steepnesses, 
		ann->cascade_activation_steepnesses_count * sizeof(fann_type));
	if(ann->cascade_activation_steepnesses == NULL)
	{
		fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
		fann_destroy(ann);
		return NULL;
	}

	fann_skip("cascade_activation_steepnesses=");
	for(i = 0; i < ann->cascade_activation_steepnesses_count; i++) 
	{
		if(fscanf(conf, FANNSCANF" ", &ann->cascade_activation_steepnesses[i]) != 1)
		{
			fann_error(NULL, FANN_E_CANT_READ_CONFIG, "cascade_activation_steepnesses", configuration_file);
			fann_destroy(ann);
			return NULL;
		}
	}

#ifdef FIXEDFANN
	ann->decimal_point = decimal_point;
	ann->multiplier = multiplier;
#endif

#ifdef FIXEDFANN
	fann_update_stepwise(ann);
#endif

#ifdef DEBUG
	printf("creating network with %d layers\n", num_layers);
	printf("input\n");
#endif

	fann_skip("layer_sizes=");
	/* determine how many neurons there should be in each layer */
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		if(fscanf(conf, "%u ", &layer_size) != 1)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_READ_CONFIG, "layer_sizes", configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		/* we do not allocate room here, but we make sure that
		 * last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		layer_it->last_neuron = layer_it->first_neuron + layer_size;
		ann->total_neurons += layer_size;
#ifdef DEBUG
		if(ann->network_type == FANN_NETTYPE_SHORTCUT && layer_it != ann->first_layer)
		{
			printf("  layer       : %d neurons, 0 bias\n", layer_size);
		}
		else
		{
			printf("  layer       : %d neurons, 1 bias\n", layer_size - 1);
		}
#endif
	}

	ann->num_input = ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1;
	ann->num_output = ((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron);
	if(ann->network_type == FANN_NETTYPE_LAYER)
	{
		/* one too many (bias) in the output layer */
		ann->num_output--;
	}

#ifndef FIXEDFANN
#define SCALE_LOAD( what, where )											\
	fann_skip( #what "_" #where "=" );									\
	for(i = 0; i < ann->num_##where##put; i++)								\
	{																		\
		if(fscanf( conf, "%f ", (float *)&ann->what##_##where[ i ] ) != 1)  \
		{																	\
			fann_error((struct fann_error *) ann, FANN_E_CANT_READ_CONFIG, #what "_" #where, configuration_file); \
			fann_destroy(ann); 												\
			return NULL;													\
		}																	\
	}
	
	if(fscanf(conf, "scale_included=%u\n", &scale_included) == 1 && scale_included == 1)
	{
		fann_allocate_scale(ann);
		SCALE_LOAD( scale_mean,			in )
		SCALE_LOAD( scale_deviation,	in )
		SCALE_LOAD( scale_new_min,		in )
		SCALE_LOAD( scale_factor,		in )
	
		SCALE_LOAD( scale_mean,			out )
		SCALE_LOAD( scale_deviation,	out )
		SCALE_LOAD( scale_new_min,		out )
		SCALE_LOAD( scale_factor,		out )
	}
#undef SCALE_LOAD
#endif
	
	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

	last_neuron = (ann->last_layer - 1)->last_neuron;
	fann_skip("neurons (num_inputs, activation_function, activation_steepness)=");
	for(neuron_it = ann->first_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		if(fscanf
		   (conf, "(%u, %u, " FANNSCANF ") ", &num_connections, &tmpVal,
			&neuron_it->activation_steepness) != 3)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_READ_NEURON, configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		neuron_it->activation_function = (enum fann_activationfunc_enum)tmpVal;
		neuron_it->first_con = ann->total_connections;
		ann->total_connections += num_connections;
		neuron_it->last_con = ann->total_connections;
	}

	fann_allocate_connections(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

	connected_neurons = ann->connections;
	weights = ann->weights;
	first_neuron = ann->first_layer->first_neuron;

	fann_skip("connections (connected_to_neuron, weight)=");
	for(i = 0; i < ann->total_connections; i++)
	{
		if(fscanf(conf, "(%u, " FANNSCANF ") ", &input_neuron, &weights[i]) != 2)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_READ_CONNECTIONS, configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		connected_neurons[i] = first_neuron + input_neuron;
	}

#ifdef DEBUG
	printf("output\n");
#endif
	return ann;
}


/* INTERNAL FUNCTION
   Create a network from a configuration file descriptor. (backward compatible read of version 1.1 files)
 */
struct fann *fann_create_from_fd_1_1(FILE * conf, const char *configuration_file)
{
	unsigned int num_layers, layer_size, input_neuron, i, network_type, num_connections;
	unsigned int activation_function_hidden, activation_function_output;
#ifdef FIXEDFANN
	unsigned int decimal_point, multiplier;
#endif
	fann_type activation_steepness_hidden, activation_steepness_output;
	float learning_rate, connection_rate;
	struct fann_neuron *first_neuron, *neuron_it, *last_neuron, **connected_neurons;
	fann_type *weights;
	struct fann_layer *layer_it;
	struct fann *ann;

#ifdef FIXEDFANN
	if(fscanf(conf, "%u\n", &decimal_point) != 1)
	{
		fann_error(NULL, FANN_E_CANT_READ_CONFIG, "decimal_point", configuration_file);
		return NULL;
	}
	multiplier = 1 << decimal_point;
#endif

	if(fscanf(conf, "%u %f %f %u %u %u " FANNSCANF " " FANNSCANF "\n", &num_layers, &learning_rate,
		&connection_rate, &network_type, &activation_function_hidden,
		&activation_function_output, &activation_steepness_hidden,
		&activation_steepness_output) != 8)
	{
		fann_error(NULL, FANN_E_CANT_READ_CONFIG, "parameters", configuration_file);
		return NULL;
	}

	ann = fann_allocate_structure(num_layers);
	if(ann == NULL)
	{
		return NULL;
	}
	ann->connection_rate = connection_rate;
	ann->network_type = (enum fann_nettype_enum)network_type;
	ann->learning_rate = learning_rate;

#ifdef FIXEDFANN
	ann->decimal_point = decimal_point;
	ann->multiplier = multiplier;
#endif

#ifdef FIXEDFANN
	fann_update_stepwise(ann);
#endif

#ifdef DEBUG
	printf("creating network with learning rate %f\n", learning_rate);
	printf("input\n");
#endif

	/* determine how many neurons there should be in each layer */
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		if(fscanf(conf, "%u ", &layer_size) != 1)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_READ_NEURON, configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		/* we do not allocate room here, but we make sure that
		 * last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		layer_it->last_neuron = layer_it->first_neuron + layer_size;
		ann->total_neurons += layer_size;
#ifdef DEBUG
		if(ann->network_type == FANN_NETTYPE_SHORTCUT && layer_it != ann->first_layer)
		{
			printf("  layer       : %d neurons, 0 bias\n", layer_size);
		}
		else
		{
			printf("  layer       : %d neurons, 1 bias\n", layer_size - 1);
		}
#endif
	}

	ann->num_input = ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1;
	ann->num_output = ((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron);
	if(ann->network_type == FANN_NETTYPE_LAYER)
	{
		/* one too many (bias) in the output layer */
		ann->num_output--;
	}

	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

	last_neuron = (ann->last_layer - 1)->last_neuron;
	for(neuron_it = ann->first_layer->first_neuron; neuron_it != last_neuron; neuron_it++)
	{
		if(fscanf(conf, "%u ", &num_connections) != 1)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_READ_NEURON, configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		neuron_it->first_con = ann->total_connections;
		ann->total_connections += num_connections;
		neuron_it->last_con = ann->total_connections;
	}

	fann_allocate_connections(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

	connected_neurons = ann->connections;
	weights = ann->weights;
	first_neuron = ann->first_layer->first_neuron;

	for(i = 0; i < ann->total_connections; i++)
	{
		if(fscanf(conf, "(%u " FANNSCANF ") ", &input_neuron, &weights[i]) != 2)
		{
			fann_error((struct fann_error *) ann, FANN_E_CANT_READ_CONNECTIONS, configuration_file);
			fann_destroy(ann);
			return NULL;
		}
		connected_neurons[i] = first_neuron + input_neuron;
	}

	fann_set_activation_steepness_hidden(ann, activation_steepness_hidden);
	fann_set_activation_steepness_output(ann, activation_steepness_output);
	fann_set_activation_function_hidden(ann, (enum fann_activationfunc_enum)activation_function_hidden);
	fann_set_activation_function_output(ann, (enum fann_activationfunc_enum)activation_function_output);

#ifdef DEBUG
	printf("output\n");
#endif
	return ann;
}
