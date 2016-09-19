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
#include <limits.h>

#include "config.h"
#include "fann.h"

#ifdef _MSC_VER
#define vsnprintf _vsnprintf
#define snprintf _snprintf
#endif

/* define path max if not defined */
#if defined(_WIN32) && !defined(__MINGW32__)
#define PATH_MAX _MAX_PATH
#endif
#ifndef PATH_MAX
#ifdef _POSIX_PATH_MAX
#define PATH_MAX _POSIX_PATH_MAX
#else
#define PATH_MAX 4096
#endif
#endif

FANN_EXTERNAL FILE * FANN_API fann_default_error_log = (FILE *)-1;

/* resets the last error number
 */
FANN_EXTERNAL void FANN_API fann_reset_errno(struct fann_error *errdat)
{
	errdat->errno_f = FANN_E_NO_ERROR;
}

/* resets the last errstr
 */
FANN_EXTERNAL void FANN_API fann_reset_errstr(struct fann_error *errdat)
{
	if(errdat->errstr != NULL)
		free(errdat->errstr);
	errdat->errstr = NULL;
}

/* returns the last error number
 */
FANN_EXTERNAL enum fann_errno_enum FANN_API fann_get_errno(struct fann_error *errdat)
{
	return errdat->errno_f;
}

/* returns the last errstr
 */
FANN_EXTERNAL char *FANN_API fann_get_errstr(struct fann_error *errdat)
{
	char *errstr = errdat->errstr;

	fann_reset_errno(errdat);
	fann_reset_errstr(errdat);

	return errstr;
}

/* change where errors are logged to
 */
FANN_EXTERNAL void FANN_API fann_set_error_log(struct fann_error *errdat, FILE * log_file)
{
	if(errdat == NULL)
		fann_default_error_log = log_file;
	else
		errdat->error_log = log_file;
}

/* prints the last error to stderr
 */
FANN_EXTERNAL void FANN_API fann_print_error(struct fann_error *errdat)
{
	if(errdat->errno_f != FANN_E_NO_ERROR && errdat->errstr != NULL)
	{
		fprintf(stderr, "FANN Error %d: %s", errdat->errno_f, errdat->errstr);
	}
}

/* INTERNAL FUNCTION
   Populate the error information
 */
void fann_error(struct fann_error *errdat, const enum fann_errno_enum errno_f, ...)
{
	va_list ap;
	size_t errstr_max = FANN_ERRSTR_MAX + PATH_MAX - 1;
	char errstr[FANN_ERRSTR_MAX + PATH_MAX];
	FILE * error_log = fann_default_error_log;

	if(errdat != NULL)
		errdat->errno_f = errno_f;

	va_start(ap, errno_f);
	switch (errno_f)
	{
	case FANN_E_NO_ERROR:
		return;
	case FANN_E_CANT_OPEN_CONFIG_R:
		vsnprintf(errstr, errstr_max, "Unable to open configuration file \"%s\" for reading.\n", ap);
		break;
	case FANN_E_CANT_OPEN_CONFIG_W:
		vsnprintf(errstr, errstr_max, "Unable to open configuration file \"%s\" for writing.\n", ap);
		break;
	case FANN_E_WRONG_CONFIG_VERSION:
		vsnprintf(errstr, errstr_max,
				 "Wrong version of configuration file, aborting read of configuration file \"%s\".\n",
				 ap);
		break;
	case FANN_E_CANT_READ_CONFIG:
		vsnprintf(errstr, errstr_max, "Error reading \"%s\" from configuration file \"%s\".\n", ap);
		break;
	case FANN_E_CANT_READ_NEURON:
		vsnprintf(errstr, errstr_max, "Error reading neuron info from configuration file \"%s\".\n", ap);
		break;
	case FANN_E_CANT_READ_CONNECTIONS:
		vsnprintf(errstr, errstr_max, "Error reading connections from configuration file \"%s\".\n", ap);
		break;
	case FANN_E_WRONG_NUM_CONNECTIONS:
		vsnprintf(errstr, errstr_max, "ERROR connections_so_far=%d, total_connections=%d\n", ap);
		break;
	case FANN_E_CANT_OPEN_TD_W:
		vsnprintf(errstr, errstr_max, "Unable to open train data file \"%s\" for writing.\n", ap);
		break;
	case FANN_E_CANT_OPEN_TD_R:
		vsnprintf(errstr, errstr_max, "Unable to open train data file \"%s\" for writing.\n", ap);
		break;
	case FANN_E_CANT_READ_TD:
		vsnprintf(errstr, errstr_max, "Error reading info from train data file \"%s\", line: %d.\n", ap);
		break;
	case FANN_E_CANT_ALLOCATE_MEM:
		strcpy(errstr, "Unable to allocate memory.\n");
		break;
	case FANN_E_CANT_TRAIN_ACTIVATION:
		strcpy(errstr, "Unable to train with the selected activation function.\n");
		break;
	case FANN_E_CANT_USE_ACTIVATION:
		strcpy(errstr, "Unable to use the selected activation function.\n");
		break;
	case FANN_E_TRAIN_DATA_MISMATCH:
		strcpy(errstr, "Training data must be of equivalent structure.\n");
		break;
	case FANN_E_CANT_USE_TRAIN_ALG:
		strcpy(errstr, "Unable to use the selected training algorithm.\n");
		break;
	case FANN_E_TRAIN_DATA_SUBSET:
		vsnprintf(errstr, errstr_max, "Subset from %d of length %d not valid in training set of length %d.\n", ap);
		break;
	case FANN_E_INDEX_OUT_OF_BOUND:
		vsnprintf(errstr, errstr_max, "Index %d is out of bound.\n", ap);
		break;
	case FANN_E_SCALE_NOT_PRESENT: 
		strcpy(errstr, "Scaling parameters not present.\n");
		break;
    case FANN_E_INPUT_NO_MATCH:
		vsnprintf(errstr, errstr_max, "The number of input neurons in the ann (%d) and data (%d) don't match\n", ap);
    	break;
    case FANN_E_OUTPUT_NO_MATCH:
		vsnprintf(errstr, errstr_max, "The number of output neurons in the ann (%d) and data (%d) don't match\n", ap);
     	break; 
	case FANN_E_WRONG_PARAMETERS_FOR_CREATE: 
		strcpy(errstr, "The parameters for create_standard are wrong, either too few parameters provided or a negative/very high value provided.\n");
		break;
	}
	va_end(ap);

	if(errdat != NULL)
	{
		if(errdat->errstr == NULL)
		{
			errdat->errstr = (char*)malloc(strlen(errstr) + 1);
		}
		else if(strlen(errdat->errstr) < strlen(errstr))
		{
			errdat->errstr = (char*)realloc(errdat->errstr, strlen(errstr) + 1);
		}
		/* allocation failed */
		if(errdat->errstr == NULL)
		{
			fprintf(stderr, "Unable to allocate memory.\n");
			return;
		}
		strcpy(errdat->errstr, errstr);
		error_log = errdat->error_log;
	}

	if(error_log == (FILE *)-1) /* This is the default behavior and will give stderr */
	{
		fprintf(stderr, "FANN Error %d: %s", errno_f, errstr);
	}
	else if(error_log != NULL)
	{
		fprintf(error_log, "FANN Error %d: %s", errno_f, errstr);
	}
}

/* INTERNAL FUNCTION
   Initialize an error data strcuture
 */
void fann_init_error_data(struct fann_error *errdat)
{
	errdat->errstr = NULL;
	errdat->errno_f = FANN_E_NO_ERROR;
	errdat->error_log = fann_default_error_log;
}
