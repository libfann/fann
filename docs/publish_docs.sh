#!/usr/bin/env bash

SCRIPT_PATH=`dirname $0`
GH_PAGES_PATH=${SCRIPT_PATH}/../../pages/fann/

${SCRIPT_PATH}/generate_docs.sh

cp -a ${SCRIPT_PATH}/html/* ${GH_PAGES_PATH}/docs/
cd ${GH_PAGES_PATH}
git commit -a -m "Automated update of docs"
git push origin gh-pages