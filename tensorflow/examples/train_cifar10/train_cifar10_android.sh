#!/bin/bash

bazel build --config=android_arm64 //tensorflow/examples/train_cifar10/... \
  --verbose_failures \
  --cxxopt="-std=c++11" \
  --cxxopt="-DSELECTIVE_REGISTRATION" \
  --cxxopt="-DSUPPORT_SELECTIVE_REGISTRATION" \
  --jobs=8

REMOTE_DIR='/data/local/tmp'

# Compiled binary code
adb push ../../../bazel-bin/tensorflow/examples/train_cifar10/train_cifar $REMOTE_DIR
adb push ../../../bazel-bin/tensorflow/examples/train_cifar10/train_and_test_cifar $REMOTE_DIR

# Send MNIST dataset to the phone. Notice that you should first create a
# cifar-10 directory using normal priviledge first
adb push ./cifar-10/data_batch_1.bin $REMOTE_DIR/cifar-10/cifar-10-batches-bin/
adb push ./cifar-10/data_batch_2.bin $REMOTE_DIR/cifar-10/cifar-10-batches-bin/
adb push ./cifar-10/data_batch_3.bin $REMOTE_DIR/cifar-10/cifar-10-batches-bin/
adb push ./cifar-10/data_batch_4.bin $REMOTE_DIR/cifar-10/cifar-10-batches-bin/
adb push ./cifar-10/test_batch.bin $REMOTE_DIR/cifar-10/cifar-10-batches-bin/

# Send the trained model: mlp/cnn network
adb push cifar10_mlp.pb $REMOTE_DIR
adb push cifar10_dnn.pb $REMOTE_DIR
