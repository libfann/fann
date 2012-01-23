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

#ifndef __fann_error_h__
#define __fann_error_h__

#include <stdio.h>

#define FANN_ERRSTR_MAX 128
struct fann_error;

/* Section: FANN Error Handling

   Errors from the fann library are usually reported on stderr. 
   It is however possible to redirect these error messages to a file, 
   or completely ignore them by the <fann_set_error_log> function.
   
   It is also possible to inspect the last error message by using the
   <fann_get_errno> and <fann_get_errstr> functions.
 */

/* Enum: fann_errno_enum
	Used to define error events on <struct fann> and <struct fann_train_data>. 

	See also:
		<fann_get_errno>, <fann_reset_errno>, <fann_get_errstr>

	FANN_E_NO_ERROR - No error 
	FANN_E_CANT_OPEN_CONFIG_R - Unable to open configuration file for reading 
	FANN_E_CANT_OPEN_CONFIG_W - Unable to open configuration file for writing
	FANN_E_WRONG_CONFIG_VERSION - Wrong version of configuration file 
	FANN_E_CANT_READ_CONFIG - Error reading info from configuration file
	FANN_E_CANT_READ_NEURON - Error reading neuron info from configuration file
	FANN_E_CANT_READ_CONNECTIONS - Error reading connections from configuration file
	FANN_E_WRONG_NUM_CONNECTIONS - Number of connections not equal to the number expected
	FANN_E_CANT_OPEN_TD_W - Unable to open train data file for writing
	FANN_E_CANT_OPEN_TD_R - Unable to open train data file for reading
	FANN_E_CANT_READ_TD - Error reading training data from file
	FANN_E_CANT_ALLOCATE_MEM - Unable to allocate memory
	FANN_E_CANT_TRAIN_ACTIVATION - Unable to train with the selected activation function
	FANN_E_CANT_USE_ACTIVATION - Unable to use the selected activation function
	FANN_E_TRAIN_DATA_MISMATCH - Irreconcilable differences between two <struct fann_train_data> structures
	FANN_E_CANT_USE_TRAIN_ALG - Unable to use the selected training algorithm
	FANN_E_TRAIN_DATA_SUBSET - Trying to take subset which is not within the training set
	FANN_E_INDEX_OUT_OF_BOUND - Index is out of bound
	FANN_E_SCALE_NOT_PRESENT - Scaling parameters not present
    FANN_E_INPUT_NO_MATCH - The number of input neurons in the ann and data don't match
    FANN_E_OUTPUT_NO_MATCH - The number of output neurons in the ann and data don't match
*/
enum fann_errno_enum
{
	FANN_E_NO_ERROR = 0,
	FANN_E_CANT_OPEN_CONFIG_R,
	FANN_E_CANT_OPEN_CONFIG_W,
	FANN_E_WRONG_CONFIG_VERSION,
	FANN_E_CANT_READ_CONFIG,
	FANN_E_CANT_READ_NEURON,
	FANN_E_CANT_READ_CONNECTIONS,
	FANN_E_WRONG_NUM_CONNECTIONS,
	FANN_E_CANT_OPEN_TD_W,
	FANN_E_CANT_OPEN_TD_R,
	FANN_E_CANT_READ_TD,
	FANN_E_CANT_ALLOCATE_MEM,
	FANN_E_CANT_TRAIN_ACTIVATION,
	FANN_E_CANT_USE_ACTIVATION,
	FANN_E_TRAIN_DATA_MISMATCH,
	FANN_E_CANT_USE_TRAIN_ALG,
	FANN_E_TRAIN_DATA_SUBSET,
	FANN_E_INDEX_OUT_OF_BOUND,
	FANN_E_SCALE_NOT_PRESENT,
	FANN_E_INPUT_NO_MATCH,
	FANN_E_OUTPUT_NO_MATCH
};

/* Group: Error Handling */
	
/* Function: fann_set_error_log

   Change where errors are logged to. Both <struct fann> and <struct fann_data> can be 
   casted to <struct fann_error>, so this function can be used to set either of these.
   
   If log_file is NULL, no errors will be printed.
   
   If errdata is NULL, the default log will be set. The default log is the log used when creating 
   <struct fann> and <struct fann_data>. This default log will also be the default for all new structs
   that are created.
   
   The default behavior is to log them to stderr.
   
   See also:
    <struct fann_error>
   
   This function appears in FANN >= 1.1.0.   
 */ 
FANN_EXTERNAL void FANN_API fann_set_error_log(struct fann_error *errdat, FILE * log_file);


/* Function: fann_get_errno

   Returns the last error number.
   
   See also:
    <fann_errno_enum>, <fann_reset_errno>
    
   This function appears in FANN >= 1.1.0.   
 */ 
FANN_EXTERNAL enum fann_errno_enum FANN_API fann_get_errno(struct fann_error *errdat);


/* Function: fann_reset_errno

   Resets the last error number.
   
   This function appears in FANN >= 1.1.0.   
 */ 
FANN_EXTERNAL void FANN_API fann_reset_errno(struct fann_error *errdat);


/* Function: fann_reset_errstr

   Resets the last error string.

   This function appears in FANN >= 1.1.0.   
 */ 
FANN_EXTERNAL void FANN_API fann_reset_errstr(struct fann_error *errdat);


/* Function: fann_get_errstr

   Returns the last errstr.
  
   This function calls <fann_reset_errno> and <fann_reset_errstr>

   This function appears in FANN >= 1.1.0.   
 */ 
FANN_EXTERNAL char *FANN_API fann_get_errstr(struct fann_error *errdat);


/* Function: fann_print_error

   Prints the last error to stderr.

   This function appears in FANN >= 1.1.0.   
 */ 
FANN_EXTERNAL void FANN_API fann_print_error(struct fann_error *errdat);

extern FILE * fann_default_error_log;

#endif
