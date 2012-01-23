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

/* Easy way to allow for build of multiple binaries */

#include "config.h"
#include "fixedfann.h"

#include "fann.c"
#include "fann_io.c"
#include "fann_train.c"
#include "fann_train_data.c"
#include "fann_error.c"
#include "fann_cascade.c"
