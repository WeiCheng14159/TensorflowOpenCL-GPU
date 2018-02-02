# TensorFlow C++ Traing MNIST tutorial

This example shows how you can train a TF AI model using TF C++ API.

## Description

This tutorial build a TF graph using python script, save the graph structure,
ship it to the phone, and feed MNIST dataset into the network.   

## Selective registration

```bash
$ bazel build tensorflow/python/tools/print_selective_registration_header
$ bazel-bin/tensorflow/python/tools/print_selective_registration_header \
  --graphs=path/to/graph.pb > ops_to_register.h
$ cp ops_to_register.h tensorflow/core/framework/ops_to_register.h
```

## Compile the program & Send the TF model to Phone

```bash
$ ./train_mnist_android.sh
```
