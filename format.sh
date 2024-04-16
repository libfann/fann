#!/bin/sh

find src -regex '.*\.\(cpp\|hpp\|c\|h\)' -exec clang-format -i {} \;
find tests -regex '.*\.\(cpp\|hpp\|c\|h\)' -exec clang-format -i {} \;
