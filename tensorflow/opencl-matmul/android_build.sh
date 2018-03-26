#!/bin/bash

TARGET=${PWD##*/}

bazel build --config=android_arm64 :$TARGET \
    --verbose_failures \
    --cxxopt="-std=c++11" \
    --cxxopt="-DSELECTIVE_REGISTRATION" \
    --cxxopt="-DSUPPORT_SELECTIVE_REGISTRATION" \
    --jobs=8

REMOTE_DIR="/data/local/tmp"
adb push ../../bazel-bin/tensorflow/$TARGET/$TARGET $REMOTE_DIR
adb push matmul.pb $REMOTE_DIR
adb push testMatmul.sh $REMOTE_DIR
adb push testSquare.sh $REMOTE_DIR
adb push testExpSquare.sh $REMOTE_DIR
