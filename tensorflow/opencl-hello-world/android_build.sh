#!/bin/bash

bazel build :opencl-hello-world \
   --verbose_failures \
   --sandbox_debug \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=arm64-v8a \
   --cxxopt="-std=c++11" \
   --jobs=4 

adb push ../../bazel-bin/tensorflow/opencl-hello-world/opencl-hello-world /data/local/tmp
