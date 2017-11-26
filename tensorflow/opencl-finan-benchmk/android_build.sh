#!/bin/bash

TARGET=opencl-monte-carlo-bench
DIR=${PWD##*/} 

bazel build :$TARGET \
   --verbose_failures \
   --sandbox_debug \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=arm64-v8a \
   --jobs=4 

adb push ../../bazel-bin/tensorflow/"$DIR"/"$TARGET" /data/local/tmp
adb push ./kernel/*cl /data/local/tmp
adb push ./Monte-Carlo/*.h /data/local/tmp

