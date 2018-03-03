#!/bin/bash

TARGET=${PWD##*/}

bazel build --config=android_arm64 :$TARGET \
   --verbose_failures \
   --cxxopt="-std=c++11" \
   --jobs=8

adb push ../../bazel-bin/tensorflow/$TARGET/$TARGET /data/local/tmp
