#!/usr/bin/env bash

SCRIPT_PATH=`dirname $0`

${SCRIPT_PATH}/NaturalDocs-1.52/NaturalDocs -r -i ${SCRIPT_PATH}/../src/include/ -o HTML ${SCRIPT_PATH}/html -p ${SCRIPT_PATH}/prj