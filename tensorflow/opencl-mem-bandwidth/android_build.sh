#!/bin/bash

bazel build --config=android_arm64 //tensorflow/opencl-mem-bandwidth/... \
   --verbose_failures \
   --cxxopt="-std=c++11" \
   --jobs=8

REMOTE_DIR="/data/local/tmp"
adb push ../../bazel-bin/tensorflow/opencl-mem-bandwidth/opencl-host_to_device $REMOTE_DIR
adb push ../../bazel-bin/tensorflow/opencl-mem-bandwidth/opencl-device_to_host $REMOTE_DIR
adb push ../../bazel-bin/tensorflow/opencl-mem-bandwidth/opencl-device_to_device $REMOTE_DIR
