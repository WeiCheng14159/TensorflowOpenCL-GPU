#!/bin/bash

TARGET=opencl-finan-black-scholes

bazel build :$TARGET \
   --verbose_failures \
   --sandbox_debug \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=arm64-v8a \
   --cxxopt="-std=c++11" \
   --jobs=4 

adb push ../../../bazel-bin/tensorflow/"$TARGET"/Black-Scholes-Benchmakr/"$TARGET" /data/local/tmp
adb push ./kernel/*cl /data/local/tmp
