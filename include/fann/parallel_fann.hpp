/*
 * parallel_FANN.hpp
 *     Author: Alessandro Pietro Bardelli
 */
#ifndef DISABLE_PARALLEL_FANN
#ifndef PARALLEL_FANN_HPP_
#define PARALLEL_FANN_HPP_
#include <omp.h>
#include <vector>
#include "fann.h"

#ifndef FIXEDFANN
namespace parallel_fann {
float train_epoch_batch_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb);

float train_epoch_irpropm_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb);

float train_epoch_quickprop_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb);

float train_epoch_sarprop_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb);

float train_epoch_incremental_mod(struct fann *ann, struct fann_train_data *data);

float train_epoch_batch_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb,std::vector< std::vector<fann_type> >& predicted_outputs);

float train_epoch_irpropm_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb, std::vector< std::vector<fann_type> >& predicted_outputs);

float train_epoch_quickprop_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb, std::vector< std::vector<fann_type> >& predicted_outputs);

float train_epoch_sarprop_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb, std::vector< std::vector<fann_type> >& predicted_outputs);

float train_epoch_incremental_mod(struct fann *ann, struct fann_train_data *data, std::vector< std::vector<fann_type> >& predicted_outputs);

float test_data_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb);
float test_data_parallel(struct fann *ann, struct fann_train_data *data, const unsigned int threadnumb, std::vector< std::vector<fann_type> >& predicted_outputs);
}
#endif /* FIXEDFANN */
#endif /* PARALLEL_FANN_HPP_ */
#endif /* DISABLE_PARALLEL_FANN */
