#!/bin/bash

bazel build :loader \
   --verbose_failures \
   --verbose_explanations\
   --sandbox_debug \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=arm64-v8a \
   --cxxopt="-std=c++11" \
   --linkopt="-pthread" \
   --copt="-DMDB_USE_ROBUST=0" \
   --jobs=4 

adb push ../../bazel-bin/tensorflow/loader/loader /data/user/0/TF
