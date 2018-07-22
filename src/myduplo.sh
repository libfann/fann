#!/bin/bash

rm -rf duplo_list.txt

#find -name "*.cpp" ! -name "moc_*.cpp" >> duplo_list.txt
find -name "*.c" >> duplo_list.txt
find -name "*.h" >> duplo_list.txt

duplo duplo_list.txt duplo_result.txt
