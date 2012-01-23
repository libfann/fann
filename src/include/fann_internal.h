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

#ifndef __fann_internal_h__
#define __fann_internal_h__
/* internal include file, not to be included directly
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "fann_data.h"

#define FANN_FIX_VERSION "FANN_FIX_2.0"
#define FANN_FLO_VERSION "FANN_FLO_2.1"

#ifdef FIXEDFANN
#define FANN_CONF_VERSION FANN_FIX_VERSION
#else
#define FANN_CONF_VERSION FANN_FLO_VERSION
#endif

#define FANN_GET(type, name) \
FANN_EXTERNAL type FANN_API fann_get_ ## name(struct fann *ann) \
{ \
	return ann->name; \
}

#define FANN_SET(type, name) \
FANN_EXTERNAL void FANN_API fann_set_ ## name(struct fann *ann, type value) \
{ \
	ann->name = value; \
}

#define FANN_GET_SET(type, name) \
FANN_GET(type, name) \
FANN_SET(type, name)


struct fann_train_data;

struct fann *fann_allocate_structure(unsigned int num_layers);
void fann_allocate_neurons(struct fann *ann);

void fann_allocate_connections(struct fann *ann);

int fann_save_internal(struct fann *ann, const char *configuration_file,
					   unsigned int save_as_fixed);
int fann_save_internal_fd(struct fann *ann, FILE * conf, const char *configuration_file,
						  unsigned int save_as_fixed);
int fann_save_train_internal(struct fann_train_data *data, const char *filename,
							  unsigned int save_as_fixed, unsigned int decimal_point);
int fann_save_train_internal_fd(struct fann_train_data *data, FILE * file, const char *filename,
								 unsigned int save_as_fixed, unsigned int decimal_point);

void fann_update_stepwise(struct fann *ann);
void fann_seed_rand();

void fann_error(struct fann_error *errdat, const enum fann_errno_enum errno_f, ...);
void fann_init_error_data(struct fann_error *errdat);

struct fann *fann_create_from_fd(FILE * conf, const char *configuration_file);
struct fann_train_data *fann_read_train_from_fd(FILE * file, const char *filename);

void fann_compute_MSE(struct fann *ann, fann_type * desired_output);
void fann_update_output_weights(struct fann *ann);
void fann_backpropagate_MSE(struct fann *ann);
void fann_update_weights(struct fann *ann);
void fann_update_slopes_batch(struct fann *ann, struct fann_layer *layer_begin,
							  struct fann_layer *layer_end);
void fann_update_weights_quickprop(struct fann *ann, unsigned int num_data,
								   unsigned int first_weight, unsigned int past_end);
void fann_update_weights_batch(struct fann *ann, unsigned int num_data, unsigned int first_weight,
							   unsigned int past_end);
void fann_update_weights_irpropm(struct fann *ann, unsigned int first_weight,
								 unsigned int past_end);
void fann_update_weights_sarprop(struct fann *ann, unsigned int epoch, unsigned int first_weight,
								unsigned int past_end);

void fann_clear_train_arrays(struct fann *ann);

fann_type fann_activation(struct fann * ann, unsigned int activation_function, fann_type steepness,
						  fann_type value);

fann_type fann_activation_derived(unsigned int activation_function,
								  fann_type steepness, fann_type value, fann_type sum);

int fann_desired_error_reached(struct fann *ann, float desired_error);

/* Some functions for cascade */
int fann_train_outputs(struct fann *ann, struct fann_train_data *data, float desired_error);

float fann_train_outputs_epoch(struct fann *ann, struct fann_train_data *data);

int fann_train_candidates(struct fann *ann, struct fann_train_data *data);

fann_type fann_train_candidates_epoch(struct fann *ann, struct fann_train_data *data);

void fann_install_candidate(struct fann *ann);
int fann_check_input_output_sizes(struct fann *ann, struct fann_train_data *data);

int fann_initialize_candidates(struct fann *ann);

void fann_set_shortcut_connections(struct fann *ann);

int fann_allocate_scale(struct fann *ann);

/* called fann_max, in order to not interferre with predefined versions of max */
#define fann_max(x, y) (((x) > (y)) ? (x) : (y))
#define fann_min(x, y) (((x) < (y)) ? (x) : (y))
#define fann_safe_free(x) {if(x) { free(x); x = NULL; }}
#define fann_clip(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))
#define fann_exp2(x) exp(0.69314718055994530942*(x))
/*#define fann_clip(x, lo, hi) (x)*/

#define fann_rand(min_value, max_value) (((float)(min_value))+(((float)(max_value)-((float)(min_value)))*rand()/(RAND_MAX+1.0f)))

#define fann_abs(value) (((value) > 0) ? (value) : -(value))

#ifdef FIXEDFANN

#define fann_mult(x,y) ((x*y) >> decimal_point)
#define fann_div(x,y) (((x) << decimal_point)/y)
#define fann_random_weight() (fann_type)(fann_rand(0,multiplier/10))
#define fann_random_bias_weight() (fann_type)(fann_rand((0-multiplier)/10,multiplier/10))

#else

#define fann_mult(x,y) (x*y)
#define fann_div(x,y) (x/y)
#define fann_random_weight() (fann_rand(-0.1f,0.1f))
#define fann_random_bias_weight() (fann_rand(-0.1f,0.1f))

#endif

#endif
