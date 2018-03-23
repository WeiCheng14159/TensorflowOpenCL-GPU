#!/bin/bash

TARGET=${PWD##*/}

# OpenCL accelerated version
bazel build --config=android_arm64 :$TARGET \
    --verbose_failures \
    --cxxopt="-std=c++11" \
    --cxxopt="-DSELECTIVE_REGISTRATION" \
    --cxxopt="-DSUPPORT_SELECTIVE_REGISTRATION" \
    --jobs=8

# CPU version
# bazel build --config=android_arm64 :$TARGET \
#     --verbose_failures \
#     --cxxopt="-std=c++11" \
#     --cxxopt="-DSELECTIVE_REGISTRATION" \
#     --cxxopt="-DSUPPORT_SELECTIVE_REGISTRATION" \
#     --jobs=8

adb push ../../bazel-bin/tensorflow/"$TARGET"/"$TARGET" /data/local/tmp
adb push models/mlp.pb /data/local/tmp
