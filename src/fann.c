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
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "config.h"
#include "fann.h"

/* #define FANN_NO_SEED */

FANN_EXTERNAL struct fann *FANN_API fann_create_standard(unsigned int num_layers, ...)
{
	struct fann *ann;
	va_list layer_sizes;
	int i;
	int status;
	int arg;
	unsigned int *layers = (unsigned int *) calloc(num_layers, sizeof(unsigned int));

	if(layers == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	va_start(layer_sizes, num_layers);
	
	status = 1;
	for(i = 0; i < (int) num_layers; i++)
	{
		arg = va_arg(layer_sizes, unsigned int);
		if(arg < 0 || arg > 1000000)
			status = 0;
		layers[i] = arg;
	}
	va_end(layer_sizes);

	if(!status)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(layers);
		return NULL;
	}

	ann = fann_create_standard_array(num_layers, layers);

	free(layers);

	return ann;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_standard_array(unsigned int num_layers, 
															   const unsigned int *layers)
{
	return fann_create_sparse_array(1, num_layers, layers);	
}

FANN_EXTERNAL struct fann *FANN_API fann_create_sparse(float connection_rate, 
													   unsigned int num_layers, ...)
{
	struct fann *ann;
	va_list layer_sizes;
	int i;
	int status;
	int arg;
	unsigned int *layers = (unsigned int *) calloc(num_layers, sizeof(unsigned int));

	if(layers == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	va_start(layer_sizes, num_layers);
	status = 1;
	for(i = 0; i < (int) num_layers; i++)
	{
		arg = va_arg(layer_sizes, unsigned int);
		if(arg < 0 || arg > 1000000)
			status = 0;
		layers[i] = arg;
	}
	va_end(layer_sizes);

	if(!status)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(layers);
		return NULL;
	}

	ann = fann_create_sparse_array(connection_rate, num_layers, layers);
	free(layers);

	return ann;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_sparse_array(float connection_rate,
															 unsigned int num_layers,
															 const unsigned int *layers)
{
	struct fann_layer *layer_it, *last_layer, *prev_layer;
	struct fann *ann;
	struct fann_neuron *neuron_it, *last_neuron, *random_neuron, *bias_neuron;
#ifdef DEBUG
	unsigned int prev_layer_size;
#endif
	unsigned int num_neurons_in, num_neurons_out, i, j;
	unsigned int min_connections, max_connections, num_connections;
	unsigned int connections_per_neuron, allocated_connections;
	unsigned int random_number, found_connection, tmp_con;

#ifdef FIXEDFANN
	unsigned int multiplier;
#endif
	if(connection_rate > 1)
	{
		connection_rate = 1;
	}

	fann_seed_rand();

	/* allocate the general structure */
	ann = fann_allocate_structure(num_layers);
	if(ann == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	ann->connection_rate = connection_rate;
#ifdef FIXEDFANN
	multiplier = ann->multiplier;
	fann_update_stepwise(ann);
#endif

	/* determine how many neurons there should be in each layer */
	i = 0;
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		/* we do not allocate room here, but we make sure that
		 * last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		layer_it->last_neuron = layer_it->first_neuron + layers[i++] + 1;	/* +1 for bias */
		ann->total_neurons += (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
	}

	ann->num_output = (unsigned int)((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron - 1);
	ann->num_input = (unsigned int)(ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);

	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

#ifdef DEBUG
	printf("creating network with connection rate %f\n", connection_rate);
	printf("input\n");
	printf("  layer       : %d neurons, 1 bias\n",
		   (int)(ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1));
#endif

	num_neurons_in = ann->num_input;
	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
	{
		num_neurons_out = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron - 1);
		/*�if all neurons in each layer should be connected to at least one neuron
		 * in the previous layer, and one neuron in the next layer.
		 * and the bias node should be connected to the all neurons in the next layer.
		 * Then this is the minimum amount of neurons */
		min_connections = fann_max(num_neurons_in, num_neurons_out); /* not calculating bias */
		max_connections = num_neurons_in * num_neurons_out;	     /* not calculating bias */
		num_connections = fann_max(min_connections,
								   (unsigned int) (0.5 + (connection_rate * max_connections))) +
								   num_neurons_out;

		connections_per_neuron = num_connections / num_neurons_out;
		allocated_connections = 0;
		/* Now split out the connections on the different neurons */
		for(i = 0; i != num_neurons_out; i++)
		{
			layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
			allocated_connections += connections_per_neuron;
			layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;

			layer_it->first_neuron[i].activation_function = FANN_SIGMOID_STEPWISE;
#ifdef FIXEDFANN
			layer_it->first_neuron[i].activation_steepness = ann->multiplier / 2;
#else
			layer_it->first_neuron[i].activation_steepness = 0.5;
#endif

			if(allocated_connections < (num_connections * (i + 1)) / num_neurons_out)
			{
				layer_it->first_neuron[i].last_con++;
				allocated_connections++;
			}
		}

		/* bias neuron also gets stuff */
		layer_it->first_neuron[i].first_con = ann->total_connections + allocated_connections;
		layer_it->first_neuron[i].last_con = ann->total_connections + allocated_connections;

		ann->total_connections += num_connections;

		/* used in the next run of the loop */
		num_neurons_in = num_neurons_out;
	}

	fann_allocate_connections(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

	if(connection_rate >= 1)
	{
#ifdef DEBUG
		prev_layer_size = ann->num_input + 1;
#endif
		prev_layer = ann->first_layer;
		last_layer = ann->last_layer;
		for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
		{
			last_neuron = layer_it->last_neuron - 1;
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
			{
				tmp_con = neuron_it->last_con - 1;
				for(i = neuron_it->first_con; i != tmp_con; i++)
				{
					ann->weights[i] = (fann_type) fann_random_weight();
					/* these connections are still initialized for fully connected networks, to allow
					 * operations to work, that are not optimized for fully connected networks.
					 */
					ann->connections[i] = prev_layer->first_neuron + (i - neuron_it->first_con);
				}

				/* bias weight */
				ann->weights[tmp_con] = (fann_type) fann_random_bias_weight();
				ann->connections[tmp_con] = prev_layer->first_neuron + (tmp_con - neuron_it->first_con);
			}
#ifdef DEBUG
			prev_layer_size = layer_it->last_neuron - layer_it->first_neuron;
#endif
			prev_layer = layer_it;
#ifdef DEBUG
			printf("  layer       : %d neurons, 1 bias\n", prev_layer_size - 1);
#endif
		}
	}
	else
	{
		/* make connections for a network, that are not fully connected */

		/* generally, what we do is first to connect all the input
		 * neurons to a output neuron, respecting the number of
		 * available input neurons for each output neuron. Then
		 * we go through all the output neurons, and connect the
		 * rest of the connections to input neurons, that they are
		 * not allready connected to.
		 */

		/* All the connections are cleared by calloc, because we want to
		 * be able to see which connections are allready connected */

		for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
		{

			num_neurons_out = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron - 1);
			num_neurons_in = (unsigned int)((layer_it - 1)->last_neuron - (layer_it - 1)->first_neuron - 1);

			/* first connect the bias neuron */
			bias_neuron = (layer_it - 1)->last_neuron - 1;
			last_neuron = layer_it->last_neuron - 1;
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
			{

				ann->connections[neuron_it->first_con] = bias_neuron;
				ann->weights[neuron_it->first_con] = (fann_type) fann_random_bias_weight();
			}

			/* then connect all neurons in the input layer */
			last_neuron = (layer_it - 1)->last_neuron - 1;
			for(neuron_it = (layer_it - 1)->first_neuron; neuron_it != last_neuron; neuron_it++)
			{

				/* random neuron in the output layer that has space
				 * for more connections */
				do
				{
					random_number = (int) (0.5 + fann_rand(0, num_neurons_out - 1));
					random_neuron = layer_it->first_neuron + random_number;
					/* checks the last space in the connections array for room */
				}
				while(ann->connections[random_neuron->last_con - 1]);

				/* find an empty space in the connection array and connect */
				for(i = random_neuron->first_con; i < random_neuron->last_con; i++)
				{
					if(ann->connections[i] == NULL)
					{
						ann->connections[i] = neuron_it;
						ann->weights[i] = (fann_type) fann_random_weight();
						break;
					}
				}
			}

			/* then connect the rest of the unconnected neurons */
			last_neuron = layer_it->last_neuron - 1;
			for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
			{
				/* find empty space in the connection array and connect */
				for(i = neuron_it->first_con; i < neuron_it->last_con; i++)
				{
					/* continue if allready connected */
					if(ann->connections[i] != NULL)
						continue;

					do
					{
						found_connection = 0;
						random_number = (int) (0.5 + fann_rand(0, num_neurons_in - 1));
						random_neuron = (layer_it - 1)->first_neuron + random_number;

						/* check to see if this connection is allready there */
						for(j = neuron_it->first_con; j < i; j++)
						{
							if(random_neuron == ann->connections[j])
							{
								found_connection = 1;
								break;
							}
						}

					}
					while(found_connection);

					/* we have found a neuron that is not allready
					 * connected to us, connect it */
					ann->connections[i] = random_neuron;
					ann->weights[i] = (fann_type) fann_random_weight();
				}
			}

#ifdef DEBUG
			printf("  layer       : %d neurons, 1 bias\n", num_neurons_out);
#endif
		}

		/* TODO it would be nice to have the randomly created
		 * connections sorted for smoother memory access.
		 */
	}

#ifdef DEBUG
	printf("output\n");
#endif

	return ann;
}


FANN_EXTERNAL struct fann *FANN_API fann_create_shortcut(unsigned int num_layers, ...)
{
	struct fann *ann;
	int i;
	int status;
	int arg;
	va_list layer_sizes;
	unsigned int *layers = (unsigned int *) calloc(num_layers, sizeof(unsigned int));

	if(layers == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	va_start(layer_sizes, num_layers);
	status = 1;
	for(i = 0; i < (int) num_layers; i++)
	{
		arg = va_arg(layer_sizes, unsigned int);
		if(arg < 0 || arg > 1000000)
			status = 0;
		layers[i] = arg;
	}
	va_end(layer_sizes);

	if(!status)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(layers);
		return NULL;
	}

	ann = fann_create_shortcut_array(num_layers, layers);

	free(layers);

	return ann;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_shortcut_array(unsigned int num_layers,
															   const unsigned int *layers)
{
	struct fann_layer *layer_it, *layer_it2, *last_layer;
	struct fann *ann;
	struct fann_neuron *neuron_it, *neuron_it2 = 0;
	unsigned int i;
	unsigned int num_neurons_in, num_neurons_out;

#ifdef FIXEDFANN
	unsigned int multiplier;
#endif
	fann_seed_rand();

	/* allocate the general structure */
	ann = fann_allocate_structure(num_layers);
	if(ann == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	ann->connection_rate = 1;
	ann->network_type = FANN_NETTYPE_SHORTCUT;
#ifdef FIXEDFANN
	multiplier = ann->multiplier;
	fann_update_stepwise(ann);
#endif

	/* determine how many neurons there should be in each layer */
	i = 0;
	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		/* we do not allocate room here, but we make sure that
		 * last_neuron - first_neuron is the number of neurons */
		layer_it->first_neuron = NULL;
		layer_it->last_neuron = layer_it->first_neuron + layers[i++];
		if(layer_it == ann->first_layer)
		{
			/* there is a bias neuron in the first layer */
			layer_it->last_neuron++;
		}

		ann->total_neurons += (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
	}

	ann->num_output = (unsigned int)((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron);
	ann->num_input = (unsigned int)(ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);

	/* allocate room for the actual neurons */
	fann_allocate_neurons(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

#ifdef DEBUG
	printf("creating fully shortcut connected network.\n");
	printf("input\n");
	printf("  layer       : %d neurons, 1 bias\n",
		   (int)(ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1));
#endif

	num_neurons_in = ann->num_input;
	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
	{
		num_neurons_out = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);

		/* Now split out the connections on the different neurons */
		for(i = 0; i != num_neurons_out; i++)
		{
			layer_it->first_neuron[i].first_con = ann->total_connections;
			ann->total_connections += num_neurons_in + 1;
			layer_it->first_neuron[i].last_con = ann->total_connections;

			layer_it->first_neuron[i].activation_function = FANN_SIGMOID_STEPWISE;
#ifdef FIXEDFANN
			layer_it->first_neuron[i].activation_steepness = ann->multiplier / 2;
#else
			layer_it->first_neuron[i].activation_steepness = 0.5;
#endif
		}

#ifdef DEBUG
		printf("  layer       : %d neurons, 0 bias\n", num_neurons_out);
#endif
		/* used in the next run of the loop */
		num_neurons_in += num_neurons_out;
	}

	fann_allocate_connections(ann);
	if(ann->errno_f == FANN_E_CANT_ALLOCATE_MEM)
	{
		fann_destroy(ann);
		return NULL;
	}

	/* Connections are created from all neurons to all neurons in later layers
	 */
	num_neurons_in = ann->num_input + 1;
	for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
	{
		for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++)
		{

			i = neuron_it->first_con;
			for(layer_it2 = ann->first_layer; layer_it2 != layer_it; layer_it2++)
			{
				for(neuron_it2 = layer_it2->first_neuron; neuron_it2 != layer_it2->last_neuron;
					neuron_it2++)
				{

					ann->weights[i] = (fann_type) fann_random_weight();
					ann->connections[i] = neuron_it2;
					i++;
				}
			}
		}
		num_neurons_in += (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
	}

#ifdef DEBUG
	printf("output\n");
#endif

	return ann;
}

FANN_EXTERNAL fann_type *FANN_API fann_run(struct fann * ann, fann_type * input)
{
	struct fann_neuron *neuron_it, *last_neuron, *neurons, **neuron_pointers;
	unsigned int i, num_connections, num_input, num_output;
	fann_type neuron_sum, *output;
	fann_type *weights;
	struct fann_layer *layer_it, *last_layer;
	unsigned int activation_function;
	fann_type steepness;

	/* store some variabels local for fast access */
	struct fann_neuron *first_neuron = ann->first_layer->first_neuron;

#ifdef FIXEDFANN
	int multiplier = ann->multiplier;
	unsigned int decimal_point = ann->decimal_point;

	/* values used for the stepwise linear sigmoid function */
	fann_type r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0;
	fann_type v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0, v6 = 0;

	fann_type last_steepness = 0;
	unsigned int last_activation_function = 0;
#else
	fann_type max_sum = 0;	
#endif

	/* first set the input */
	num_input = ann->num_input;
	for(i = 0; i != num_input; i++)
	{
#ifdef FIXEDFANN
		if(fann_abs(input[i]) > multiplier)
		{
			printf
				("Warning input number %d is out of range -%d - %d with value %d, integer overflow may occur.\n",
				 i, multiplier, multiplier, input[i]);
		}
#endif
		first_neuron[i].value = input[i];
	}
	/* Set the bias neuron in the input layer */
#ifdef FIXEDFANN
	(ann->first_layer->last_neuron - 1)->value = multiplier;
#else
	(ann->first_layer->last_neuron - 1)->value = 1;
#endif

	last_layer = ann->last_layer;
	for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;
		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			if(neuron_it->first_con == neuron_it->last_con)
			{
				/* bias neurons */
#ifdef FIXEDFANN
				neuron_it->value = multiplier;
#else
				neuron_it->value = 1;
#endif
				continue;
			}

			activation_function = neuron_it->activation_function;
			steepness = neuron_it->activation_steepness;

			neuron_sum = 0;
			num_connections = neuron_it->last_con - neuron_it->first_con;
			weights = ann->weights + neuron_it->first_con;

			if(ann->connection_rate >= 1)
			{
				if(ann->network_type == FANN_NETTYPE_SHORTCUT)
				{
					neurons = ann->first_layer->first_neuron;
				}
				else
				{
					neurons = (layer_it - 1)->first_neuron;
				}


				/* unrolled loop start */
				i = num_connections & 3;	/* same as modulo 4 */
				switch (i)
				{
					case 3:
						neuron_sum += fann_mult(weights[2], neurons[2].value);
					case 2:
						neuron_sum += fann_mult(weights[1], neurons[1].value);
					case 1:
						neuron_sum += fann_mult(weights[0], neurons[0].value);
					case 0:
						break;
				}

				for(; i != num_connections; i += 4)
				{
					neuron_sum +=
						fann_mult(weights[i], neurons[i].value) +
						fann_mult(weights[i + 1], neurons[i + 1].value) +
						fann_mult(weights[i + 2], neurons[i + 2].value) +
						fann_mult(weights[i + 3], neurons[i + 3].value);
				}
				/* unrolled loop end */

				/*
				 * for(i = 0;i != num_connections; i++){
				 * printf("%f += %f*%f, ", neuron_sum, weights[i], neurons[i].value);
				 * neuron_sum += fann_mult(weights[i], neurons[i].value);
				 * }
				 */
			}
			else
			{
				neuron_pointers = ann->connections + neuron_it->first_con;

				i = num_connections & 3;	/* same as modulo 4 */
				switch (i)
				{
					case 3:
						neuron_sum += fann_mult(weights[2], neuron_pointers[2]->value);
					case 2:
						neuron_sum += fann_mult(weights[1], neuron_pointers[1]->value);
					case 1:
						neuron_sum += fann_mult(weights[0], neuron_pointers[0]->value);
					case 0:
						break;
				}

				for(; i != num_connections; i += 4)
				{
					neuron_sum +=
						fann_mult(weights[i], neuron_pointers[i]->value) +
						fann_mult(weights[i + 1], neuron_pointers[i + 1]->value) +
						fann_mult(weights[i + 2], neuron_pointers[i + 2]->value) +
						fann_mult(weights[i + 3], neuron_pointers[i + 3]->value);
				}
			}

#ifdef FIXEDFANN
			neuron_it->sum = fann_mult(steepness, neuron_sum);

			if(activation_function != last_activation_function || steepness != last_steepness)
			{
				switch (activation_function)
				{
					case FANN_SIGMOID:
					case FANN_SIGMOID_STEPWISE:
						r1 = ann->sigmoid_results[0];
						r2 = ann->sigmoid_results[1];
						r3 = ann->sigmoid_results[2];
						r4 = ann->sigmoid_results[3];
						r5 = ann->sigmoid_results[4];
						r6 = ann->sigmoid_results[5];
						v1 = ann->sigmoid_values[0] / steepness;
						v2 = ann->sigmoid_values[1] / steepness;
						v3 = ann->sigmoid_values[2] / steepness;
						v4 = ann->sigmoid_values[3] / steepness;
						v5 = ann->sigmoid_values[4] / steepness;
						v6 = ann->sigmoid_values[5] / steepness;
						break;
					case FANN_SIGMOID_SYMMETRIC:
					case FANN_SIGMOID_SYMMETRIC_STEPWISE:
						r1 = ann->sigmoid_symmetric_results[0];
						r2 = ann->sigmoid_symmetric_results[1];
						r3 = ann->sigmoid_symmetric_results[2];
						r4 = ann->sigmoid_symmetric_results[3];
						r5 = ann->sigmoid_symmetric_results[4];
						r6 = ann->sigmoid_symmetric_results[5];
						v1 = ann->sigmoid_symmetric_values[0] / steepness;
						v2 = ann->sigmoid_symmetric_values[1] / steepness;
						v3 = ann->sigmoid_symmetric_values[2] / steepness;
						v4 = ann->sigmoid_symmetric_values[3] / steepness;
						v5 = ann->sigmoid_symmetric_values[4] / steepness;
						v6 = ann->sigmoid_symmetric_values[5] / steepness;
						break;
					case FANN_THRESHOLD:
						break;
				}
			}

			switch (activation_function)
			{
				case FANN_SIGMOID:
				case FANN_SIGMOID_STEPWISE:
					neuron_it->value =
						(fann_type) fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, 0,
												  multiplier, neuron_sum);
					break;
				case FANN_SIGMOID_SYMMETRIC:
				case FANN_SIGMOID_SYMMETRIC_STEPWISE:
					neuron_it->value =
						(fann_type) fann_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6,
												  -multiplier, multiplier, neuron_sum);
					break;
				case FANN_THRESHOLD:
					neuron_it->value = (fann_type) ((neuron_sum < 0) ? 0 : multiplier);
					break;
				case FANN_THRESHOLD_SYMMETRIC:
					neuron_it->value = (fann_type) ((neuron_sum < 0) ? -multiplier : multiplier);
					break;
				case FANN_LINEAR:
					neuron_it->value = neuron_sum;
					break;
				case FANN_LINEAR_PIECE:
					neuron_it->value = (fann_type)((neuron_sum < 0) ? 0 : (neuron_sum > multiplier) ? multiplier : neuron_sum);
					break;
				case FANN_LINEAR_PIECE_SYMMETRIC:
					neuron_it->value = (fann_type)((neuron_sum < -multiplier) ? -multiplier : (neuron_sum > multiplier) ? multiplier : neuron_sum);
					break;
				case FANN_ELLIOT:
				case FANN_ELLIOT_SYMMETRIC:
				case FANN_GAUSSIAN:
				case FANN_GAUSSIAN_SYMMETRIC:
				case FANN_GAUSSIAN_STEPWISE:
				case FANN_SIN_SYMMETRIC:
				case FANN_COS_SYMMETRIC:
					fann_error((struct fann_error *) ann, FANN_E_CANT_USE_ACTIVATION);
					break;
			}
			last_steepness = steepness;
			last_activation_function = activation_function;
#else
			neuron_sum = fann_mult(steepness, neuron_sum);
			
			max_sum = 150/steepness;
			if(neuron_sum > max_sum)
				neuron_sum = max_sum;
			else if(neuron_sum < -max_sum)
				neuron_sum = -max_sum;
			
			neuron_it->sum = neuron_sum;

			fann_activation_switch(activation_function, neuron_sum, neuron_it->value);
#endif
		}
	}

	/* set the output */
	output = ann->output;
	num_output = ann->num_output;
	neurons = (ann->last_layer - 1)->first_neuron;
	for(i = 0; i != num_output; i++)
	{
		output[i] = neurons[i].value;
	}
	return ann->output;
}

FANN_EXTERNAL void FANN_API fann_destroy(struct fann *ann)
{
	if(ann == NULL)
		return;
	fann_safe_free(ann->weights);
	fann_safe_free(ann->connections);
	fann_safe_free(ann->first_layer->first_neuron);
	fann_safe_free(ann->first_layer);
	fann_safe_free(ann->output);
	fann_safe_free(ann->train_errors);
	fann_safe_free(ann->train_slopes);
	fann_safe_free(ann->prev_train_slopes);
	fann_safe_free(ann->prev_steps);
	fann_safe_free(ann->prev_weights_deltas);
	fann_safe_free(ann->errstr);
	fann_safe_free(ann->cascade_activation_functions);
	fann_safe_free(ann->cascade_activation_steepnesses);
	fann_safe_free(ann->cascade_candidate_scores);
	
#ifndef FIXEDFANN
	fann_safe_free( ann->scale_mean_in );
	fann_safe_free( ann->scale_deviation_in );
	fann_safe_free( ann->scale_new_min_in );
	fann_safe_free( ann->scale_factor_in );

	fann_safe_free( ann->scale_mean_out );
	fann_safe_free( ann->scale_deviation_out );
	fann_safe_free( ann->scale_new_min_out );
	fann_safe_free( ann->scale_factor_out );
#endif
	
	fann_safe_free(ann);
}

FANN_EXTERNAL void FANN_API fann_randomize_weights(struct fann *ann, fann_type min_weight,
												   fann_type max_weight)
{
	fann_type *last_weight;
	fann_type *weights = ann->weights;

	last_weight = weights + ann->total_connections;
	for(; weights != last_weight; weights++)
	{
		*weights = (fann_type) (fann_rand(min_weight, max_weight));
	}

#ifndef FIXEDFANN
	if(ann->prev_train_slopes != NULL)
	{
		fann_clear_train_arrays(ann);
	}
#endif
}

/* deep copy of the fann structure */
FANN_EXTERNAL struct fann* FANN_API fann_copy(struct fann* orig)
{
    struct fann* copy;
    unsigned int num_layers = (unsigned int)(orig->last_layer - orig->first_layer);
    struct fann_layer *orig_layer_it, *copy_layer_it;
    unsigned int layer_size;
    struct fann_neuron *last_neuron,*orig_neuron_it,*copy_neuron_it;
    unsigned int i;
    struct fann_neuron *orig_first_neuron,*copy_first_neuron;
    unsigned int input_neuron;

    copy = fann_allocate_structure(num_layers);
    if (copy==NULL) {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
    copy->errno_f = orig->errno_f;
    if (orig->errstr)
    {
        copy->errstr = (char *) malloc(FANN_ERRSTR_MAX);
        if (copy->errstr == NULL)
        {
            fann_destroy(copy);
            return NULL;
        }
        strcpy(copy->errstr,orig->errstr);
    }
    copy->error_log = orig->error_log;

    copy->learning_rate = orig->learning_rate;
    copy->learning_momentum = orig->learning_momentum;
    copy->connection_rate = orig->connection_rate;
    copy->network_type = orig->network_type;
    copy->num_MSE = orig->num_MSE;
    copy->MSE_value = orig->MSE_value;
    copy->num_bit_fail = orig->num_bit_fail;
    copy->bit_fail_limit = orig->bit_fail_limit;
    copy->train_error_function = orig->train_error_function;
    copy->train_stop_function = orig->train_stop_function;
	copy->training_algorithm = orig->training_algorithm;
    copy->callback = orig->callback;
	copy->user_data = orig->user_data;
#ifndef FIXEDFANN
    copy->cascade_output_change_fraction = orig->cascade_output_change_fraction;
    copy->cascade_output_stagnation_epochs = orig->cascade_output_stagnation_epochs;
    copy->cascade_candidate_change_fraction = orig->cascade_candidate_change_fraction;
    copy->cascade_candidate_stagnation_epochs = orig->cascade_candidate_stagnation_epochs;
    copy->cascade_best_candidate = orig->cascade_best_candidate;
    copy->cascade_candidate_limit = orig->cascade_candidate_limit;
    copy->cascade_weight_multiplier = orig->cascade_weight_multiplier;
    copy->cascade_max_out_epochs = orig->cascade_max_out_epochs;
    copy->cascade_max_cand_epochs = orig->cascade_max_cand_epochs;

   /* copy cascade activation functions */
    copy->cascade_activation_functions_count = orig->cascade_activation_functions_count;
    copy->cascade_activation_functions = (enum fann_activationfunc_enum *)realloc(copy->cascade_activation_functions,
        copy->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
    if(copy->cascade_activation_functions == NULL)
    {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy(copy);
        return NULL;
    }
    memcpy(copy->cascade_activation_functions,orig->cascade_activation_functions,
            copy->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));

    /* copy cascade activation steepnesses */
    copy->cascade_activation_steepnesses_count = orig->cascade_activation_steepnesses_count;
    copy->cascade_activation_steepnesses = (fann_type *)realloc(copy->cascade_activation_steepnesses, copy->cascade_activation_steepnesses_count * sizeof(fann_type));
    if(copy->cascade_activation_steepnesses == NULL)
    {
        fann_error((struct fann_error*)orig, FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy(copy);
        return NULL;
    }
    memcpy(copy->cascade_activation_steepnesses,orig->cascade_activation_steepnesses,copy->cascade_activation_steepnesses_count * sizeof(fann_type));

    copy->cascade_num_candidate_groups = orig->cascade_num_candidate_groups;

    /* copy candidate scores, if used */
    if (orig->cascade_candidate_scores == NULL)
    {
        copy->cascade_candidate_scores = NULL;
    }
    else
    {
        copy->cascade_candidate_scores =
            (fann_type *) malloc(fann_get_cascade_num_candidates(copy) * sizeof(fann_type));
        if(copy->cascade_candidate_scores == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->cascade_candidate_scores,orig->cascade_candidate_scores,fann_get_cascade_num_candidates(copy) * sizeof(fann_type));
    }
#endif /* FIXEDFANN */

    copy->quickprop_decay = orig->quickprop_decay;
    copy->quickprop_mu = orig->quickprop_mu;
    copy->rprop_increase_factor = orig->rprop_increase_factor;
    copy->rprop_decrease_factor = orig->rprop_decrease_factor;
    copy->rprop_delta_min = orig->rprop_delta_min;
    copy->rprop_delta_max = orig->rprop_delta_max;
    copy->rprop_delta_zero = orig->rprop_delta_zero;

    /* user_data is not deep copied.  user should use fann_copy_with_user_data() for that */
    copy->user_data = orig->user_data;

#ifdef FIXEDFANN
    copy->decimal_point = orig->decimal_point;
    copy->multiplier = orig->multiplier;
    memcpy(copy->sigmoid_results,orig->sigmoid_results,6*sizeof(fann_type));
    memcpy(copy->sigmoid_values,orig->sigmoid_values,6*sizeof(fann_type));
    memcpy(copy->sigmoid_symmetric_results,orig->sigmoid_symmetric_results,6*sizeof(fann_type));
    memcpy(copy->sigmoid_symmetric_values,orig->sigmoid_symmetric_values,6*sizeof(fann_type));
#endif


    /* copy layer sizes, prepare for fann_allocate_neurons */
    for (orig_layer_it = orig->first_layer, copy_layer_it = copy->first_layer;
            orig_layer_it != orig->last_layer; orig_layer_it++, copy_layer_it++)
    {
        layer_size = (unsigned int)(orig_layer_it->last_neuron - orig_layer_it->first_neuron);
        copy_layer_it->first_neuron = NULL;
        copy_layer_it->last_neuron = copy_layer_it->first_neuron + layer_size;
        copy->total_neurons += layer_size;
    }
    copy->num_input = orig->num_input;
    copy->num_output = orig->num_output;


    /* copy scale parameters, when used */
#ifndef FIXEDFANN
    if (orig->scale_mean_in != NULL)
    {
        fann_allocate_scale(copy);
        for (i=0; i < orig->num_input ; i++) {
            copy->scale_mean_in[i] = orig->scale_mean_in[i];
            copy->scale_deviation_in[i] = orig->scale_deviation_in[i];
            copy->scale_new_min_in[i] = orig->scale_new_min_in[i];
            copy->scale_factor_in[i] = orig->scale_factor_in[i];
        }
        for (i=0; i < orig->num_output ; i++) {
            copy->scale_mean_out[i] = orig->scale_mean_out[i];
            copy->scale_deviation_out[i] = orig->scale_deviation_out[i];
            copy->scale_new_min_out[i] = orig->scale_new_min_out[i];
            copy->scale_factor_out[i] = orig->scale_factor_out[i];
        }
    }
#endif

    /* copy the neurons */
    fann_allocate_neurons(copy);
    if (copy->errno_f == FANN_E_CANT_ALLOCATE_MEM)
    {
        fann_destroy(copy);
        return NULL;
    }
    layer_size = (unsigned int)((orig->last_layer-1)->last_neuron - (orig->last_layer-1)->first_neuron);
    memcpy(copy->output,orig->output, layer_size * sizeof(fann_type));

    last_neuron = (orig->last_layer - 1)->last_neuron;
    for (orig_neuron_it = orig->first_layer->first_neuron, copy_neuron_it = copy->first_layer->first_neuron;
            orig_neuron_it != last_neuron; orig_neuron_it++, copy_neuron_it++)
    {
        memcpy(copy_neuron_it,orig_neuron_it,sizeof(struct fann_neuron));
    }
 /* copy the connections */
    copy->total_connections = orig->total_connections;
    fann_allocate_connections(copy);
    if (copy->errno_f == FANN_E_CANT_ALLOCATE_MEM)
    {
        fann_destroy(copy);
        return NULL;
    }

    orig_first_neuron = orig->first_layer->first_neuron;
    copy_first_neuron = copy->first_layer->first_neuron;
    for (i=0; i < orig->total_connections; i++)
    {
        copy->weights[i] = orig->weights[i];
        input_neuron = (unsigned int)(orig->connections[i] - orig_first_neuron);
        copy->connections[i] = copy_first_neuron + input_neuron;
    }

    if (orig->train_slopes)
    {
        copy->train_slopes = (fann_type *) malloc(copy->total_connections_allocated * sizeof(fann_type));
        if (copy->train_slopes == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->train_slopes,orig->train_slopes,copy->total_connections_allocated * sizeof(fann_type));
    }

    if (orig->prev_steps)
    {
        copy->prev_steps = (fann_type *) malloc(copy->total_connections_allocated * sizeof(fann_type));
        if (copy->prev_steps == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->prev_steps, orig->prev_steps, copy->total_connections_allocated * sizeof(fann_type));
    }

    if (orig->prev_train_slopes)
    {
        copy->prev_train_slopes = (fann_type *) malloc(copy->total_connections_allocated * sizeof(fann_type));
        if (copy->prev_train_slopes == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->prev_train_slopes,orig->prev_train_slopes, copy->total_connections_allocated * sizeof(fann_type));
    }

    if (orig->prev_weights_deltas)
    {
        copy->prev_weights_deltas = (fann_type *) malloc(copy->total_connections_allocated * sizeof(fann_type));
        if(copy->prev_weights_deltas == NULL)
        {
            fann_error((struct fann_error *) orig, FANN_E_CANT_ALLOCATE_MEM);
            fann_destroy(copy);
            return NULL;
        }
        memcpy(copy->prev_weights_deltas, orig->prev_weights_deltas,copy->total_connections_allocated * sizeof(fann_type));
    }

    return copy;
}

FANN_EXTERNAL void FANN_API fann_print_connections(struct fann *ann)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it;
	unsigned int i;
	int value;
	char *neurons;
	unsigned int num_neurons = fann_get_total_neurons(ann) - fann_get_num_output(ann);

	neurons = (char *) malloc(num_neurons + 1);
	if(neurons == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	neurons[num_neurons] = 0;

	printf("Layer / Neuron ");
	for(i = 0; i < num_neurons; i++)
	{
		printf("%d", i % 10);
	}
	printf("\n");

	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
	{
		for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++)
		{

			memset(neurons, (int) '.', num_neurons);
			for(i = neuron_it->first_con; i < neuron_it->last_con; i++)
			{
				if(ann->weights[i] < 0)
				{
#ifdef FIXEDFANN
					value = (int) ((ann->weights[i] / (double) ann->multiplier) - 0.5);
#else
					value = (int) ((ann->weights[i]) - 0.5);
#endif
					if(value < -25)
						value = -25;
					neurons[ann->connections[i] - ann->first_layer->first_neuron] = (char)('a' - value);
				}
				else
				{
#ifdef FIXEDFANN
					value = (int) ((ann->weights[i] / (double) ann->multiplier) + 0.5);
#else
					value = (int) ((ann->weights[i]) + 0.5);
#endif
					if(value > 25)
						value = 25;
					neurons[ann->connections[i] - ann->first_layer->first_neuron] = (char)('A' + value);
				}
			}
			printf("L %3d / N %4d %s\n", (int)(layer_it - ann->first_layer),
				   (int)(neuron_it - ann->first_layer->first_neuron), neurons);
		}
	}

	free(neurons);
}

/* Initialize the weights using Widrow + Nguyen's algorithm.
*/
FANN_EXTERNAL void FANN_API fann_init_weights(struct fann *ann, struct fann_train_data *train_data)
{
	fann_type smallest_inp, largest_inp;
	unsigned int dat = 0, elem, num_connect, num_hidden_neurons;
	struct fann_layer *layer_it;
	struct fann_neuron *neuron_it, *last_neuron, *bias_neuron;

#ifdef FIXEDFANN
	unsigned int multiplier = ann->multiplier;
#endif
	float scale_factor;

	for(smallest_inp = largest_inp = train_data->input[0][0]; dat < train_data->num_data; dat++)
	{
		for(elem = 0; elem < train_data->num_input; elem++)
		{
			if(train_data->input[dat][elem] < smallest_inp)
				smallest_inp = train_data->input[dat][elem];
			if(train_data->input[dat][elem] > largest_inp)
				largest_inp = train_data->input[dat][elem];
		}
	}

	num_hidden_neurons = (unsigned int)(
		ann->total_neurons - (ann->num_input + ann->num_output +
							  (ann->last_layer - ann->first_layer)));
	scale_factor =
		(float) (pow
				 ((double) (0.7f * (double) num_hidden_neurons),
				  (double) (1.0f / (double) ann->num_input)) / (double) (largest_inp -
																		 smallest_inp));

#ifdef DEBUG
	printf("Initializing weights with scale factor %f\n", scale_factor);
#endif
	bias_neuron = ann->first_layer->last_neuron - 1;
	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++)
	{
		last_neuron = layer_it->last_neuron;

		if(ann->network_type == FANN_NETTYPE_LAYER)
		{
			bias_neuron = (layer_it - 1)->last_neuron - 1;
		}

		for(neuron_it = layer_it->first_neuron; neuron_it != last_neuron; neuron_it++)
		{
			for(num_connect = neuron_it->first_con; num_connect < neuron_it->last_con;
				num_connect++)
			{
				if(bias_neuron == ann->connections[num_connect])
				{
#ifdef FIXEDFANN
					ann->weights[num_connect] =
						(fann_type) fann_rand(-scale_factor, scale_factor * multiplier);
#else
					ann->weights[num_connect] = (fann_type) fann_rand(-scale_factor, scale_factor);
#endif
				}
				else
				{
#ifdef FIXEDFANN
					ann->weights[num_connect] = (fann_type) fann_rand(0, scale_factor * multiplier);
#else
					ann->weights[num_connect] = (fann_type) fann_rand(0, scale_factor);
#endif
				}
			}
		}
	}

#ifndef FIXEDFANN
	if(ann->prev_train_slopes != NULL)
	{
		fann_clear_train_arrays(ann);
	}
#endif
}

FANN_EXTERNAL void FANN_API fann_print_parameters(struct fann *ann)
{
	struct fann_layer *layer_it;
#ifndef FIXEDFANN
	unsigned int i;
#endif

	printf("Input layer                          :%4d neurons, 1 bias\n", ann->num_input);
	for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer - 1; layer_it++)
	{
		if(ann->network_type == FANN_NETTYPE_SHORTCUT)
		{
			printf("  Hidden layer                       :%4d neurons, 0 bias\n",
				   (int)(layer_it->last_neuron - layer_it->first_neuron));
		}
		else
		{
			printf("  Hidden layer                       :%4d neurons, 1 bias\n",
				   (int)(layer_it->last_neuron - layer_it->first_neuron - 1));
		}
	}
	printf("Output layer                         :%4d neurons\n", ann->num_output);
	printf("Total neurons and biases             :%4d\n", fann_get_total_neurons(ann));
	printf("Total connections                    :%4d\n", ann->total_connections);
	printf("Connection rate                      :%8.3f\n", ann->connection_rate);
	printf("Network type                         :   %s\n", FANN_NETTYPE_NAMES[ann->network_type]);
#ifdef FIXEDFANN
	printf("Decimal point                        :%4d\n", ann->decimal_point);
	printf("Multiplier                           :%4d\n", ann->multiplier);
#else
	printf("Training algorithm                   :   %s\n", FANN_TRAIN_NAMES[ann->training_algorithm]);
	printf("Training error function              :   %s\n", FANN_ERRORFUNC_NAMES[ann->train_error_function]);
	printf("Training stop function               :   %s\n", FANN_STOPFUNC_NAMES[ann->train_stop_function]);
#endif
#ifdef FIXEDFANN
	printf("Bit fail limit                       :%4d\n", ann->bit_fail_limit);
#else
	printf("Bit fail limit                       :%8.3f\n", ann->bit_fail_limit);
	printf("Learning rate                        :%8.3f\n", ann->learning_rate);
	printf("Learning momentum                    :%8.3f\n", ann->learning_momentum);
	printf("Quickprop decay                      :%11.6f\n", ann->quickprop_decay);
	printf("Quickprop mu                         :%8.3f\n", ann->quickprop_mu);
	printf("RPROP increase factor                :%8.3f\n", ann->rprop_increase_factor);
	printf("RPROP decrease factor                :%8.3f\n", ann->rprop_decrease_factor);
	printf("RPROP delta min                      :%8.3f\n", ann->rprop_delta_min);
	printf("RPROP delta max                      :%8.3f\n", ann->rprop_delta_max);
	printf("Cascade output change fraction       :%11.6f\n", ann->cascade_output_change_fraction);
	printf("Cascade candidate change fraction    :%11.6f\n", ann->cascade_candidate_change_fraction);
	printf("Cascade output stagnation epochs     :%4d\n", ann->cascade_output_stagnation_epochs);
	printf("Cascade candidate stagnation epochs  :%4d\n", ann->cascade_candidate_stagnation_epochs);
	printf("Cascade max output epochs            :%4d\n", ann->cascade_max_out_epochs);
	printf("Cascade min output epochs            :%4d\n", ann->cascade_min_out_epochs);
	printf("Cascade max candidate epochs         :%4d\n", ann->cascade_max_cand_epochs);
	printf("Cascade min candidate epochs         :%4d\n", ann->cascade_min_cand_epochs);
	printf("Cascade weight multiplier            :%8.3f\n", ann->cascade_weight_multiplier);
	printf("Cascade candidate limit              :%8.3f\n", ann->cascade_candidate_limit);
	for(i = 0; i < ann->cascade_activation_functions_count; i++)
		printf("Cascade activation functions[%d]      :   %s\n", i,
			FANN_ACTIVATIONFUNC_NAMES[ann->cascade_activation_functions[i]]);
	for(i = 0; i < ann->cascade_activation_steepnesses_count; i++)
		printf("Cascade activation steepnesses[%d]    :%8.3f\n", i,
			ann->cascade_activation_steepnesses[i]);
		
	printf("Cascade candidate groups             :%4d\n", ann->cascade_num_candidate_groups);
	printf("Cascade no. of candidates            :%4d\n", fann_get_cascade_num_candidates(ann));
	
	/* TODO: dump scale parameters */
#endif
}

FANN_GET(unsigned int, num_input)
FANN_GET(unsigned int, num_output)

FANN_EXTERNAL unsigned int FANN_API fann_get_total_neurons(struct fann *ann)
{
	if(ann->network_type)
	{
		return ann->total_neurons;
	}
	else
	{
		/* -1, because there is always an unused bias neuron in the last layer */
		return ann->total_neurons - 1;
	}
}

FANN_GET(unsigned int, total_connections)

FANN_EXTERNAL enum fann_nettype_enum FANN_API fann_get_network_type(struct fann *ann)
{
    /* Currently two types: LAYER = 0, SHORTCUT = 1 */
    /* Enum network_types must be set to match the return values  */
    return ann->network_type;
}

FANN_EXTERNAL float FANN_API fann_get_connection_rate(struct fann *ann)
{
    return ann->connection_rate;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_layers(struct fann *ann)
{
    return (unsigned int)(ann->last_layer - ann->first_layer);
}

FANN_EXTERNAL void FANN_API fann_get_layer_array(struct fann *ann, unsigned int *layers)
{
    struct fann_layer *layer_it;

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
        unsigned int count = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
        /* Remove the bias from the count of neurons. */
        switch (fann_get_network_type(ann)) {
            case FANN_NETTYPE_LAYER: {
                --count;
                break;
            }
            case FANN_NETTYPE_SHORTCUT: {
                /* The bias in the first layer is reused for all layers */
                if (layer_it == ann->first_layer)
                    --count;
                break;
            }
            default: {
                /* Unknown network type, assume no bias present  */
                break;
            }
        }
        *layers++ = count;
    }
}

FANN_EXTERNAL void FANN_API fann_get_bias_array(struct fann *ann, unsigned int *bias)
{
    struct fann_layer *layer_it;

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; ++layer_it, ++bias) {
        switch (fann_get_network_type(ann)) {
            case FANN_NETTYPE_LAYER: {
                /* Report one bias in each layer except the last */
                if (layer_it != ann->last_layer-1)
                    *bias = 1;
                else
                    *bias = 0;
                break;
            }
            case FANN_NETTYPE_SHORTCUT: {
                /* The bias in the first layer is reused for all layers */
                if (layer_it == ann->first_layer)
                    *bias = 1;
                else
                    *bias = 0;
                break;
            }
            default: {
                /* Unknown network type, assume no bias present  */
                *bias = 0;
                break;
            }
        }
    }
}

FANN_EXTERNAL void FANN_API fann_get_connection_array(struct fann *ann, struct fann_connection *connections)
{
    struct fann_neuron *first_neuron;
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    unsigned int idx;
    unsigned int source_index;
    unsigned int destination_index;

    first_neuron = ann->first_layer->first_neuron;

    source_index = 0;
    destination_index = 0;
    
    /* The following assumes that the last unused bias has no connections */

    /* for each layer */
    for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
        /* for each neuron */
        for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
            /* for each connection */
            for (idx = neuron_it->first_con; idx < neuron_it->last_con; idx++){
                /* Assign the source, destination and weight */
                connections->from_neuron = (unsigned int)(ann->connections[source_index] - first_neuron);
                connections->to_neuron = destination_index;
                connections->weight = ann->weights[source_index];

                connections++;
                source_index++;
            }
            destination_index++;
        }
    }
}

FANN_EXTERNAL void FANN_API fann_set_weight_array(struct fann *ann,
    struct fann_connection *connections, unsigned int num_connections)
{
    unsigned int idx;

    for (idx = 0; idx < num_connections; idx++) {
        fann_set_weight(ann, connections[idx].from_neuron,
            connections[idx].to_neuron, connections[idx].weight);
    }
}

FANN_EXTERNAL void FANN_API fann_set_weight(struct fann *ann,
    unsigned int from_neuron, unsigned int to_neuron, fann_type weight)
{
    struct fann_neuron *first_neuron;
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    unsigned int idx;
    unsigned int source_index;
    unsigned int destination_index;

    first_neuron = ann->first_layer->first_neuron;

    source_index = 0;
    destination_index = 0;

    /* Find the connection, simple brute force search through the network
       for one or more connections that match to minimize datastructure dependencies.
       Nothing is done if the connection does not already exist in the network. */

    /* for each layer */
    for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++){
        /* for each neuron */
        for(neuron_it = layer_it->first_neuron; neuron_it != layer_it->last_neuron; neuron_it++){
            /* for each connection */
            for (idx = neuron_it->first_con; idx < neuron_it->last_con; idx++){
                /* If the source and destination neurons match, assign the weight */
                if (((int)from_neuron == ann->connections[source_index] - first_neuron) &&
                    (to_neuron == destination_index))
                {
                    ann->weights[source_index] = weight;
                }
                source_index++;
            }
            destination_index++;
        }
    }
}

FANN_EXTERNAL void FANN_API fann_get_weights(struct fann *ann, fann_type *weights)
{
	memcpy(weights, ann->weights, sizeof(fann_type)*ann->total_connections);
}

FANN_EXTERNAL void FANN_API fann_set_weights(struct fann *ann, fann_type *weights)
{
	memcpy(ann->weights, weights, sizeof(fann_type)*ann->total_connections);
}

FANN_GET_SET(void *, user_data)

#ifdef FIXEDFANN

FANN_GET(unsigned int, decimal_point)
FANN_GET(unsigned int, multiplier)

/* INTERNAL FUNCTION
   Adjust the steepwise functions (if used)
*/
void fann_update_stepwise(struct fann *ann)
{
	unsigned int i = 0;

	/* Calculate the parameters for the stepwise linear
	 * sigmoid function fixed point.
	 * Using a rewritten sigmoid function.
	 * results 0.005, 0.05, 0.25, 0.75, 0.95, 0.995
	 */
	ann->sigmoid_results[0] = fann_max((fann_type) (ann->multiplier / 200.0 + 0.5), 1);
	ann->sigmoid_results[1] = fann_max((fann_type) (ann->multiplier / 20.0 + 0.5), 1);
	ann->sigmoid_results[2] = fann_max((fann_type) (ann->multiplier / 4.0 + 0.5), 1);
	ann->sigmoid_results[3] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 4.0 + 0.5), ann->multiplier - 1);
	ann->sigmoid_results[4] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 20.0 + 0.5), ann->multiplier - 1);
	ann->sigmoid_results[5] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 200.0 + 0.5), ann->multiplier - 1);

	ann->sigmoid_symmetric_results[0] = fann_max((fann_type) ((ann->multiplier / 100.0) - ann->multiplier - 0.5),
				                                 (fann_type) (1 - (fann_type) ann->multiplier));
	ann->sigmoid_symmetric_results[1] =	fann_max((fann_type) ((ann->multiplier / 10.0) - ann->multiplier - 0.5),
				                                 (fann_type) (1 - (fann_type) ann->multiplier));
	ann->sigmoid_symmetric_results[2] =	fann_max((fann_type) ((ann->multiplier / 2.0) - ann->multiplier - 0.5),
                                				 (fann_type) (1 - (fann_type) ann->multiplier));
	ann->sigmoid_symmetric_results[3] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 2.0 + 0.5),
				 							     ann->multiplier - 1);
	ann->sigmoid_symmetric_results[4] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 10.0 + 0.5),
				 							     ann->multiplier - 1);
	ann->sigmoid_symmetric_results[5] = fann_min(ann->multiplier - (fann_type) (ann->multiplier / 100.0 + 1.0),
				 							     ann->multiplier - 1);

	for(i = 0; i < 6; i++)
	{
		ann->sigmoid_values[i] =
			(fann_type) (((log(ann->multiplier / (float) ann->sigmoid_results[i] - 1) *
						   (float) ann->multiplier) / -2.0) * (float) ann->multiplier);
		ann->sigmoid_symmetric_values[i] =
			(fann_type) (((log
						   ((ann->multiplier -
							 (float) ann->sigmoid_symmetric_results[i]) /
							((float) ann->sigmoid_symmetric_results[i] +
							 ann->multiplier)) * (float) ann->multiplier) / -2.0) *
						 (float) ann->multiplier);
	}
}
#endif


/* INTERNAL FUNCTION
   Allocates the main structure and sets some default values.
 */
struct fann *fann_allocate_structure(unsigned int num_layers)
{
	struct fann *ann;

	if(num_layers < 2)
	{
#ifdef DEBUG
		printf("less than 2 layers - ABORTING.\n");
#endif
		return NULL;
	}

	/* allocate and initialize the main network structure */
	ann = (struct fann *) malloc(sizeof(struct fann));
	if(ann == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		return NULL;
	}

	ann->errno_f = FANN_E_NO_ERROR;
	ann->error_log = fann_default_error_log;
	ann->errstr = NULL;
	ann->learning_rate = 0.7f;
	ann->learning_momentum = 0.0;
	ann->total_neurons = 0;
	ann->total_connections = 0;
	ann->num_input = 0;
	ann->num_output = 0;
	ann->train_errors = NULL;
	ann->train_slopes = NULL;
	ann->prev_steps = NULL;
	ann->prev_train_slopes = NULL;
	ann->prev_weights_deltas = NULL;
	ann->training_algorithm = FANN_TRAIN_RPROP;
	ann->num_MSE = 0;
	ann->MSE_value = 0;
	ann->num_bit_fail = 0;
	ann->bit_fail_limit = (fann_type)0.35;
	ann->network_type = FANN_NETTYPE_LAYER;
	ann->train_error_function = FANN_ERRORFUNC_TANH;
	ann->train_stop_function = FANN_STOPFUNC_MSE;
	ann->callback = NULL;
    ann->user_data = NULL; /* User is responsible for deallocation */
	ann->weights = NULL;
	ann->connections = NULL;
	ann->output = NULL;
#ifndef FIXEDFANN
	ann->scale_mean_in = NULL;
	ann->scale_deviation_in = NULL;
	ann->scale_new_min_in = NULL;
	ann->scale_factor_in = NULL;
	ann->scale_mean_out = NULL;
	ann->scale_deviation_out = NULL;
	ann->scale_new_min_out = NULL;
	ann->scale_factor_out = NULL;
#endif	
	
	/* variables used for cascade correlation (reasonable defaults) */
	ann->cascade_output_change_fraction = 0.01f;
	ann->cascade_candidate_change_fraction = 0.01f;
	ann->cascade_output_stagnation_epochs = 12;
	ann->cascade_candidate_stagnation_epochs = 12;
	ann->cascade_num_candidate_groups = 2;
	ann->cascade_weight_multiplier = (fann_type)0.4;
	ann->cascade_candidate_limit = (fann_type)1000.0;
	ann->cascade_max_out_epochs = 150;
	ann->cascade_max_cand_epochs = 150;
	ann->cascade_min_out_epochs = 50;
	ann->cascade_min_cand_epochs = 50;
	ann->cascade_candidate_scores = NULL;
	ann->cascade_activation_functions_count = 10;
	ann->cascade_activation_functions = 
		(enum fann_activationfunc_enum *)calloc(ann->cascade_activation_functions_count, 
							   sizeof(enum fann_activationfunc_enum));
	if(ann->cascade_activation_functions == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(ann);
		return NULL;
	}
							   
	ann->cascade_activation_functions[0] = FANN_SIGMOID;
	ann->cascade_activation_functions[1] = FANN_SIGMOID_SYMMETRIC;
	ann->cascade_activation_functions[2] = FANN_GAUSSIAN;
	ann->cascade_activation_functions[3] = FANN_GAUSSIAN_SYMMETRIC;
	ann->cascade_activation_functions[4] = FANN_ELLIOT;
	ann->cascade_activation_functions[5] = FANN_ELLIOT_SYMMETRIC;
	ann->cascade_activation_functions[6] = FANN_SIN_SYMMETRIC;
	ann->cascade_activation_functions[7] = FANN_COS_SYMMETRIC;
	ann->cascade_activation_functions[8] = FANN_SIN;
	ann->cascade_activation_functions[9] = FANN_COS;

	ann->cascade_activation_steepnesses_count = 4;
	ann->cascade_activation_steepnesses = 
		(fann_type *)calloc(ann->cascade_activation_steepnesses_count, 
							   sizeof(fann_type));
	if(ann->cascade_activation_steepnesses == NULL)
	{
		fann_safe_free(ann->cascade_activation_functions);
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(ann);
		return NULL;
	}
	
	ann->cascade_activation_steepnesses[0] = (fann_type)0.25;
	ann->cascade_activation_steepnesses[1] = (fann_type)0.5;
	ann->cascade_activation_steepnesses[2] = (fann_type)0.75;
	ann->cascade_activation_steepnesses[3] = (fann_type)1.0;

	/* Variables for use with with Quickprop training (reasonable defaults) */
	ann->quickprop_decay = -0.0001f;
	ann->quickprop_mu = 1.75;

	/* Variables for use with with RPROP training (reasonable defaults) */
	ann->rprop_increase_factor = 1.2f;
	ann->rprop_decrease_factor = 0.5;
	ann->rprop_delta_min = 0.0;
	ann->rprop_delta_max = 50.0;
	ann->rprop_delta_zero = 0.1f;
	
 	/* Variables for use with SARPROP training (reasonable defaults) */
 	ann->sarprop_weight_decay_shift = -6.644f;
 	ann->sarprop_step_error_threshold_factor = 0.1f;
 	ann->sarprop_step_error_shift = 1.385f;
 	ann->sarprop_temperature = 0.015f;
 	ann->sarprop_epoch = 0;
 
	fann_init_error_data((struct fann_error *) ann);

#ifdef FIXEDFANN
	/* these values are only boring defaults, and should really
	 * never be used, since the real values are always loaded from a file. */
	ann->decimal_point = 8;
	ann->multiplier = 256;
#endif

	/* allocate room for the layers */
	ann->first_layer = (struct fann_layer *) calloc(num_layers, sizeof(struct fann_layer));
	if(ann->first_layer == NULL)
	{
		fann_error(NULL, FANN_E_CANT_ALLOCATE_MEM);
		free(ann);
		return NULL;
	}

	ann->last_layer = ann->first_layer + num_layers;

	return ann;
}

/* INTERNAL FUNCTION
   Allocates room for the scaling parameters.
 */
int fann_allocate_scale(struct fann *ann)
{
	/* todo this should only be allocated when needed */
#ifndef FIXEDFANN
	unsigned int i = 0;
#define SCALE_ALLOCATE( what, where, default_value )		    			\
		ann->what##_##where = (float *)calloc(								\
			ann->num_##where##put,											\
			sizeof( float )													\
			);																\
		if( ann->what##_##where == NULL )									\
		{																	\
			fann_error( NULL, FANN_E_CANT_ALLOCATE_MEM );					\
			fann_destroy( ann );                            				\
			return 1;														\
		}																	\
		for( i = 0; i < ann->num_##where##put; i++ )						\
			ann->what##_##where[ i ] = ( default_value );

	SCALE_ALLOCATE( scale_mean,		in,		0.0 )
	SCALE_ALLOCATE( scale_deviation,	in,		1.0 )
	SCALE_ALLOCATE( scale_new_min,	in,		-1.0 )
	SCALE_ALLOCATE( scale_factor,		in,		1.0 )

	SCALE_ALLOCATE( scale_mean,		out,	0.0 )
	SCALE_ALLOCATE( scale_deviation,	out,	1.0 )
	SCALE_ALLOCATE( scale_new_min,	out,	-1.0 )
	SCALE_ALLOCATE( scale_factor,		out,	1.0 )
#undef SCALE_ALLOCATE
#endif	
	return 0;
}

/* INTERNAL FUNCTION
   Allocates room for the neurons.
 */
void fann_allocate_neurons(struct fann *ann)
{
	struct fann_layer *layer_it;
	struct fann_neuron *neurons;
	unsigned int num_neurons_so_far = 0;
	unsigned int num_neurons = 0;

	/* all the neurons is allocated in one long array (calloc clears mem) */
	neurons = (struct fann_neuron *) calloc(ann->total_neurons, sizeof(struct fann_neuron));
	ann->total_neurons_allocated = ann->total_neurons;

	if(neurons == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}

	for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
	{
		num_neurons = (unsigned int)(layer_it->last_neuron - layer_it->first_neuron);
		layer_it->first_neuron = neurons + num_neurons_so_far;
		layer_it->last_neuron = layer_it->first_neuron + num_neurons;
		num_neurons_so_far += num_neurons;
	}

	ann->output = (fann_type *) calloc(num_neurons, sizeof(fann_type));
	if(ann->output == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
}

/* INTERNAL FUNCTION
   Allocate room for the connections.
 */
void fann_allocate_connections(struct fann *ann)
{
	ann->weights = (fann_type *) calloc(ann->total_connections, sizeof(fann_type));
	if(ann->weights == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
	ann->total_connections_allocated = ann->total_connections;

	/* TODO make special cases for all places where the connections
	 * is used, so that it is not needed for fully connected networks.
	 */
	ann->connections =
		(struct fann_neuron **) calloc(ann->total_connections_allocated,
									   sizeof(struct fann_neuron *));
	if(ann->connections == NULL)
	{
		fann_error((struct fann_error *) ann, FANN_E_CANT_ALLOCATE_MEM);
		return;
	}
}

#ifdef FANN_NO_SEED
int FANN_SEED_RAND = 0;
#else
int FANN_SEED_RAND = 1;
#endif

FANN_EXTERNAL void FANN_API fann_disable_seed_rand()
{
    FANN_SEED_RAND = 0;
}

FANN_EXTERNAL void FANN_API fann_enable_seed_rand()
{
    FANN_SEED_RAND = 1;
}

/* INTERNAL FUNCTION
   Seed the random function.
 */
void fann_seed_rand()
{
#ifndef _WIN32
	FILE *fp = fopen("/dev/urandom", "r");
	unsigned int foo;
	struct timeval t;

	if(!fp)
	{
		gettimeofday(&t, NULL);
		foo = t.tv_usec;
#ifdef DEBUG
		printf("unable to open /dev/urandom\n");
#endif
	}
	else
	{
	        if(fread(&foo, sizeof(foo), 1, fp) != 1) 
	        {
  		       gettimeofday(&t, NULL);
		       foo = t.tv_usec;
#ifdef DEBUG
		       printf("unable to read from /dev/urandom\n");
#endif		      
		}
		fclose(fp);
	}
    if(FANN_SEED_RAND) {
        srand(foo);
    }
#else
	/* COMPAT_TIME REPLACEMENT */
    if(FANN_SEED_RAND) {
    	srand(GetTickCount());
    }
#endif
}

