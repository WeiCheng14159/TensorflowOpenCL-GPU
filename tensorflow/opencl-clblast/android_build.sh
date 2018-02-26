#!/bin/bash

bazel build --config=android_arm64 //tensorflow/opencl-clblast/... \
   --verbose_failures \
   --cxxopt="-std=c++11" \
   --cxxopt="-DOPENCL_API" \
   --jobs=8

REMOTE_DIR="/data/local/tmp"
adb push ../../bazel-bin/tensorflow/opencl-clblast/opencl-sgemm_batched $REMOTE_DIR
adb push ../../bazel-bin/tensorflow/opencl-clblast/opencl-sgemm $REMOTE_DIR
adb push ../../bazel-bin/tensorflow/opencl-clblast/opencl-hgemm $REMOTE_DIR
adb push ../../bazel-bin/tensorflow/opencl-clblast/opencl-sgemm-tuned $REMOTE_DIR
adb push ../../bazel-bin/tensorflow/opencl-clblast/opencl-hgemm-tuned $REMOTE_DIR
