/*
 * parallel_FANN.c
 *     Author: Alessandro Pietro Bardelli
 */
#ifndef DISABLE_PARALLEL_FANN
#include <omp.h>
#include "parallel_fann.h"
#include "config.h"
#include "fann.h"

FANN_EXTERNAL float FANN_API fann_train_epoch_batch_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb)
{
	/*vector<struct fann *> ann_vect(threadnumb);*/
	struct fann** ann_vect= (struct fann**) malloc(threadnumb * sizeof(struct fann*));
	int i=0,j=0;
	fann_reset_MSE(ann);

	//generate copies of the ann
	omp_set_dynamic(0);
	omp_set_num_threads(threadnumb);
	#pragma omp parallel private(j)
	{

		#pragma omp for schedule(static)
		for(i=0; i<(int)threadnumb; i++)
		{
			ann_vect[i]=fann_copy(ann);
		}

    //parallel computing of the updates

        #pragma omp for schedule(static)
		for(i = 0; i < (int)data->num_data; i++)
		{
			j=omp_get_thread_num();
			fann_run(ann_vect[j], data->input[i]);
			fann_compute_MSE(ann_vect[j], data->output[i]);
			fann_backpropagate_MSE(ann_vect[j]);
			fann_update_slopes_batch(ann_vect[j], ann_vect[j]->first_layer + 1, ann_vect[j]->last_layer - 1);
		}
	}

    //parallel update of the weights
	{
		const unsigned int num_data=data->num_data;
		const unsigned int first_weight=0;
		const unsigned int past_end=ann->total_connections;
		fann_type *weights = ann->weights;
		const fann_type epsilon = ann->learning_rate / num_data;
		omp_set_dynamic(0);
		omp_set_num_threads(threadnumb);
		#pragma omp parallel
		{
			#pragma omp for schedule(static)
				for(i=first_weight; i < (int)past_end; i++)
				{
					fann_type temp_slopes=0.0;
					unsigned int k;
					fann_type *train_slopes;
					for(k=0;k<threadnumb;++k)
					{
						train_slopes=ann_vect[k]->train_slopes;
						temp_slopes+= train_slopes[i];
						train_slopes[i]=0.0;
					}
					weights[i] += temp_slopes*epsilon;
				}
			}
	}
	//merge of MSEs
	for(i=0;i<(int)threadnumb;++i)
	{
		ann->MSE_value+= ann_vect[i]->MSE_value;
		ann->num_MSE+=ann_vect[i]->num_MSE;
		fann_destroy(ann_vect[i]);
	}
	free(ann_vect);
	return fann_get_MSE(ann);
}


FANN_EXTERNAL float FANN_API fann_train_epoch_irpropm_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb)
{
	struct fann** ann_vect= (struct fann**) malloc(threadnumb * sizeof(struct fann*));
	int i=0,j=0;

	if(ann->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	//#define THREADNUM 1
	fann_reset_MSE(ann);

	/*vector<struct fann *> ann_vect(threadnumb);*/

	//generate copies of the ann
	omp_set_dynamic(0);
	omp_set_num_threads(threadnumb);
	#pragma omp parallel private(j)
	{

		#pragma omp for schedule(static)
		for(i=0; i<(int)threadnumb; i++)
		{
			ann_vect[i]=fann_copy(ann);
		}

    //parallel computing of the updates


        #pragma omp for schedule(static)
		for(i = 0; i < (int)data->num_data; i++)
		{
			j=omp_get_thread_num();
			fann_run(ann_vect[j], data->input[i]);
			fann_compute_MSE(ann_vect[j], data->output[i]);
			fann_backpropagate_MSE(ann_vect[j]);
			fann_update_slopes_batch(ann_vect[j], ann_vect[j]->first_layer + 1, ann_vect[j]->last_layer - 1);
		}
	}

	{
    	fann_type *weights = ann->weights;
    	fann_type *prev_steps = ann->prev_steps;
    	fann_type *prev_train_slopes = ann->prev_train_slopes;

    	fann_type next_step;

    	const float increase_factor = ann->rprop_increase_factor;	//1.2;
    	const float decrease_factor = ann->rprop_decrease_factor;	//0.5;
    	const float delta_min = ann->rprop_delta_min;	//0.0;
    	const float delta_max = ann->rprop_delta_max;	//50.0;
		const unsigned int first_weight=0;
		const unsigned int past_end=ann->total_connections;

		omp_set_dynamic(0);
		omp_set_num_threads(threadnumb);
		#pragma omp parallel private(next_step)
		{
			#pragma omp for schedule(static)
				for(i=first_weight; i < (int)past_end; i++)
				{
					fann_type prev_slope, same_sign;
		    		const fann_type prev_step = fann_max(prev_steps[i], (fann_type) 0.0001);	// prev_step may not be zero because then the training will stop

		    		fann_type temp_slopes=0.0;
					unsigned int k;
					fann_type *train_slopes;
					for(k=0;k<threadnumb;++k)
					{
						train_slopes=ann_vect[k]->train_slopes;
						temp_slopes+= train_slopes[i];
						train_slopes[i]=0.0;
					}

		    		prev_slope = prev_train_slopes[i];

		    		same_sign = prev_slope * temp_slopes;

		    		if(same_sign >= 0.0)
		    			next_step = fann_min(prev_step * increase_factor, delta_max);
		    		else
		    		{
		    			next_step = fann_max(prev_step * decrease_factor, delta_min);
		    			temp_slopes = 0;
		    		}

		    		if(temp_slopes < 0)
		    		{
		    			weights[i] -= next_step;
		    			if(weights[i] < -1500)
		    				weights[i] = -1500;
		    		}
		    		else
		    		{
		    			weights[i] += next_step;
		    			if(weights[i] > 1500)
		    				weights[i] = 1500;
		    		}

		    		// update global data arrays
		    		prev_steps[i] = next_step;
		    		prev_train_slopes[i] = temp_slopes;

				}
			}
	}

	//merge of MSEs
	for(i=0;i<(int)threadnumb;++i)
	{
		ann->MSE_value+= ann_vect[i]->MSE_value;
		ann->num_MSE+=ann_vect[i]->num_MSE;
		fann_destroy(ann_vect[i]);
	}
	free(ann_vect);
	return fann_get_MSE(ann);
}


FANN_EXTERNAL float FANN_API fann_train_epoch_quickprop_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb)
{
	struct fann** ann_vect= (struct fann**) malloc(threadnumb * sizeof(struct fann*));
	int i=0,j=0;

	if(ann->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	//#define THREADNUM 1
	fann_reset_MSE(ann);

	/*vector<struct fann *> ann_vect(threadnumb);*/

	//generate copies of the ann
	omp_set_dynamic(0);
	omp_set_num_threads(threadnumb);
	#pragma omp parallel private(j)
	{

		#pragma omp for schedule(static)
		for(i=0; i<(int)threadnumb; i++)
		{
			ann_vect[i]=fann_copy(ann);
		}

    //parallel computing of the updates

        #pragma omp for schedule(static)
		for(i = 0; i < (int)data->num_data; i++)
		{
			j=omp_get_thread_num();
			fann_run(ann_vect[j], data->input[i]);
			fann_compute_MSE(ann_vect[j], data->output[i]);
			fann_backpropagate_MSE(ann_vect[j]);
			fann_update_slopes_batch(ann_vect[j], ann_vect[j]->first_layer + 1, ann_vect[j]->last_layer - 1);
		}
	}

    {
    	fann_type *weights = ann->weights;
    	fann_type *prev_steps = ann->prev_steps;
    	fann_type *prev_train_slopes = ann->prev_train_slopes;
		const unsigned int first_weight=0;
		const unsigned int past_end=ann->total_connections;

    	fann_type w=0.0, next_step;

    	const float epsilon = ann->learning_rate / data->num_data;
    	const float decay = ann->quickprop_decay;	/*-0.0001;*/
    	const float mu = ann->quickprop_mu;	/*1.75; */
    	const float shrink_factor = (float) (mu / (1.0 + mu));

		omp_set_dynamic(0);
		omp_set_num_threads(threadnumb);
		#pragma omp parallel private(w, next_step)
		{
			#pragma omp for schedule(static)
				for(i=first_weight; i < (int)past_end; i++)
				{
					fann_type temp_slopes=0.0;
					unsigned int k;
					fann_type *train_slopes;
					fann_type prev_step, prev_slope;

					w = weights[i];
					for(k=0;k<threadnumb;++k)
					{
						train_slopes=ann_vect[k]->train_slopes;
						temp_slopes+= train_slopes[i];
						train_slopes[i]=0.0;
					}
					temp_slopes+= decay * w;

					prev_step = prev_steps[i];
					prev_slope = prev_train_slopes[i];

					next_step = 0.0;


					/* The step must always be in direction opposite to the slope. */
					if(prev_step > 0.001)
					{
						/* If last step was positive...  */
						if(temp_slopes > 0.0) /*  Add in linear term if current slope is still positive. */
							next_step += epsilon * temp_slopes;

						/*If current slope is close to or larger than prev slope...  */
						if(temp_slopes > (shrink_factor * prev_slope))
							next_step += mu * prev_step;	/* Take maximum size negative step. */
						else
							next_step += prev_step * temp_slopes / (prev_slope - temp_slopes);	/* Else, use quadratic estimate. */
					}
					else if(prev_step < -0.001)
					{
						/* If last step was negative...  */
						if(temp_slopes < 0.0) /*  Add in linear term if current slope is still negative. */
							next_step += epsilon * temp_slopes;

						/* If current slope is close to or more neg than prev slope... */
						if(temp_slopes < (shrink_factor * prev_slope))
							next_step += mu * prev_step;	/* Take maximum size negative step. */
						else
							next_step += prev_step * temp_slopes / (prev_slope - temp_slopes);	/* Else, use quadratic estimate. */
					}
					else /* Last step was zero, so use only linear term. */
						next_step += epsilon * temp_slopes;

					/* update global data arrays */
					prev_steps[i] = next_step;
					prev_train_slopes[i] = temp_slopes;

					w += next_step;

					if(w > 1500)
						weights[i] = 1500;
					else if(w < -1500)
						weights[i] = -1500;
					else
						weights[i] = w;
				}
		}
	}
	//merge of MSEs
	for(i=0;i<(int)threadnumb;++i)
	{
		ann->MSE_value+= ann_vect[i]->MSE_value;
		ann->num_MSE+=ann_vect[i]->num_MSE;
		fann_destroy(ann_vect[i]);
	}
	free(ann_vect);
	return fann_get_MSE(ann);
}


FANN_EXTERNAL float FANN_API fann_train_epoch_sarprop_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb)
{
	struct fann** ann_vect= (struct fann**) malloc(threadnumb * sizeof(struct fann*));
	int i=0,j=0;

	if(ann->prev_train_slopes == NULL)
	{
		fann_clear_train_arrays(ann);
	}

	//#define THREADNUM 1
	fann_reset_MSE(ann);

	/*vector<struct fann *> ann_vect(threadnumb);*/

	//generate copies of the ann
	omp_set_dynamic(0);
	omp_set_num_threads(threadnumb);
	#pragma omp parallel private(j)
	{

		#pragma omp for schedule(static)
		for(i=0; i<(int)threadnumb; i++)
		{
			ann_vect[i]=fann_copy(ann);
		}

    //parallel computing of the updates

        #pragma omp for schedule(static)
		for(i = 0; i < (int)data->num_data; i++)
		{
			j=omp_get_thread_num();
			fann_run(ann_vect[j], data->input[i]);
			fann_compute_MSE(ann_vect[j], data->output[i]);
			fann_backpropagate_MSE(ann_vect[j]);
			fann_update_slopes_batch(ann_vect[j], ann_vect[j]->first_layer + 1, ann_vect[j]->last_layer - 1);
		}
	}

    {
    	fann_type *weights = ann->weights;
    	fann_type *prev_steps = ann->prev_steps;
    	fann_type *prev_train_slopes = ann->prev_train_slopes;
		const unsigned int first_weight=0;
		const unsigned int past_end=ann->total_connections;
		const unsigned int epoch=ann->sarprop_epoch;

    	fann_type next_step;

    	/* These should be set from variables */
    	const float increase_factor = ann->rprop_increase_factor;	/*1.2; */
    	const float decrease_factor = ann->rprop_decrease_factor;	/*0.5; */
    	/* TODO: why is delta_min 0.0 in iRprop? SARPROP uses 1x10^-6 (Braun and Riedmiller, 1993) */
    	const float delta_min = 0.000001f;
    	const float delta_max = ann->rprop_delta_max;	/*50.0; */
    	const float weight_decay_shift = ann->sarprop_weight_decay_shift; /* ld 0.01 = -6.644 */
    	const float step_error_threshold_factor = ann->sarprop_step_error_threshold_factor; /* 0.1 */
    	const float step_error_shift = ann->sarprop_step_error_shift; /* ld 3 = 1.585 */
    	const float T = ann->sarprop_temperature;
		float MSE, RMSE;


    	//merge of MSEs
    	for(i=0;i<(int)threadnumb;++i)
    	{
    		ann->MSE_value+= ann_vect[i]->MSE_value;
    		ann->num_MSE+=ann_vect[i]->num_MSE;
    	}

    	MSE = fann_get_MSE(ann);
    	RMSE = sqrtf(MSE);

    	/* for all weights; TODO: are biases included? */
		omp_set_dynamic(0);
		omp_set_num_threads(threadnumb);
		#pragma omp parallel private(next_step)
		{
			#pragma omp for schedule(static)
				for(i=first_weight; i < (int)past_end; i++)
				{
					/* TODO: confirm whether 1x10^-6 == delta_min is really better */
					const fann_type prev_step  = fann_max(prev_steps[i], (fann_type) 0.000001);	/* prev_step may not be zero because then the training will stop */

					/* calculate SARPROP slope; TODO: better as new error function? (see SARPROP paper)*/
					fann_type prev_slope, same_sign;
					fann_type temp_slopes=0.0;
					unsigned int k;
					fann_type *train_slopes;
					for(k=0;k<threadnumb;++k)
					{
						train_slopes=ann_vect[k]->train_slopes;
						temp_slopes+= train_slopes[i];
						train_slopes[i]=0.0;
					}
					temp_slopes= -temp_slopes - weights[i] * (fann_type)fann_exp2(-T * epoch + weight_decay_shift);

					next_step=0.0;

					/* TODO: is prev_train_slopes[i] 0.0 in the beginning? */
					prev_slope = prev_train_slopes[i];

					same_sign = prev_slope * temp_slopes;

					if(same_sign > 0.0)
					{
						next_step = fann_min(prev_step * increase_factor, delta_max);
						/* TODO: are the signs inverted? see differences between SARPROP paper and iRprop */
						if (temp_slopes < 0.0)
							weights[i] += next_step;
						else
							weights[i] -= next_step;
					}
					else if(same_sign < 0.0)
					{
						#ifndef RAND_MAX
						#define	RAND_MAX	0x7fffffff
						#endif
						if(prev_step < step_error_threshold_factor * MSE)
							next_step = prev_step * decrease_factor + (float)rand() / RAND_MAX * RMSE * (fann_type)fann_exp2(-T * epoch + step_error_shift);
						else
							next_step = fann_max(prev_step * decrease_factor, delta_min);

						temp_slopes = 0.0;
					}
					else
					{
						if(temp_slopes < 0.0)
							weights[i] += prev_step;
						else
							weights[i] -= prev_step;
					}

					/* update global data arrays */
					prev_steps[i] = next_step;
					prev_train_slopes[i] = temp_slopes;

				}
		}
    }

	++(ann->sarprop_epoch);

	//already computed before
	/*//merge of MSEs
	for(i=0;i<threadnumb;++i)
	{
		ann->MSE_value+= ann_vect[i]->MSE_value;
		ann->num_MSE+=ann_vect[i]->num_MSE;
	}*/
	//destroy the copies of the ann
	for(i=0; i<(int)threadnumb; i++)
	{
		fann_destroy(ann_vect[i]);
	}
	free(ann_vect);
	return fann_get_MSE(ann);
}

FANN_EXTERNAL float FANN_API fann_train_epoch_incremental_mod(struct fann *ann, struct fann_train_data *data)
{
	unsigned int i;

	fann_reset_MSE(ann);

	for(i = 0; i != data->num_data; i++)
	{
		fann_train(ann, data->input[i], data->output[i]);
	}

	return fann_get_MSE(ann);
}

#endif /* DISABLE_PARALLEL_FANN */
