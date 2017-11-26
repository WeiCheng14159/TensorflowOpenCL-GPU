#!/bin/bash

TARGETS=`ls src | cut -d'.' -f 1`
PROJ_DIR=${PWD##*/}

TF_NEED_OPENCL_SYCL=1
TF_NEED_COMPUTECPP=0

#TARGETS="example_sycl_app"

for TARGET in $TARGETS
do  
    echo "compiling $TARGET" 
    bazel build :$TARGET \
       --verbose_failures \
       --sandbox_debug \
       --crosstool_top=//external:android/crosstool \
       --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
       --cpu=arm64-v8a \
       --cxxopt="-std=c++11" \
       --jobs=4 
        
        adb push ../../bazel-bin/tensorflow/$PROJ_DIR/$TARGET /data/local/tmp
done 

