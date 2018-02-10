#!/bin/bash

TARGET=${PWD##*/}

bazel build --config=android_arm64 :$TARGET \
   --verbose_failures \
   --jobs=8

REMOTE_DIR=/data/local/tmp

adb push ../../bazel-bin/tensorflow/$TARGET/$TARGET $REMOTE_DIR
adb push kernels/*.cl $REMOTE_DIR
