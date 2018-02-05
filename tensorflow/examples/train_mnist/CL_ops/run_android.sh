#!/bin/bash

bazel build :zero_out_op_kernel_1.so \
    --verbose_failures=true \
    --cxxopt="-ferror-limit=1" \
    --jobs=8

bazel build --config=android_arm64 :zero_out_op_kernel_1.so \
    --verbose_failures=true \
    --cxxopt="-ferror-limit=1" \
    --cxxopt="-std=c++11" \
    --jobs=8

REMOTE_DIR='/data/local/tmp'

# Compiled binary code
# adb push ../../../bazel-bin/tensorflow/examples/CL_Ops/attr_examples $REMOTE_DIR
