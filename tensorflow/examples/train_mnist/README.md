# Training the MNIST model with TensorFlow C++ API

This example trains a TF AI model using TF C++ API.

## Description

The key idea is as follow:

Build a TF model using Python script

Save the model

Ship it to the phone

Feed the MNIST dataset into the network.   

## Selective registration

TensorFlow only supports inference on mobile platform for now, and the
training feature requires some Ops not included in `//tensorflow/core:android_tensorflow_lib`.

By selective registration, all Ops
needed by a TF model will be generated into a file ops_to_register.h.

Move the `ops_to_register.h ` file to `tensorflow/core/framework/ops_to_register.h`
and add the following compiler flags `--cxxopt="-DSELECTIVE_REGISTRATION" --cxxopt="-DSUPPORT_SELECTIVE_REGISTRATION" ` to the Bazel compile command.

The Ops needed will be added into the `//tensorflow/core:android_tensorflow_lib`


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
