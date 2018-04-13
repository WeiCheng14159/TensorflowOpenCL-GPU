# Description:
#   TensorFlow C++ training example for CIFAR10 dataset

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

ANDROID_C_OPTS = tf_copts() + [
    "-ffunction-sections",
    "-fdata-sections",
    "-fPIE",
    "-pie",
    "-fexceptions",
]

ANDROID_LINK_OPTS = [
    "-fPIE",
    "-pie",
    "-landroid",
    "-latomic",
    "-ldl",
    "-llog",
    "-lm",
    "-z defs",
    "-s",
    "-Wl,--gc-sections",
    "-fuse-ld=gold",
]

cc_library(
    name = "cifar10Reader",
    srcs = [
        "cifar10Reader.cc",
    ],
    hdrs = [
        "cifar10Reader.h",
    ],
    deps = [
        "//external:libcifar10",
        "//tensorflow/core:android_tensorflow_lib",
    ],
)

cc_library(
    name = "tf_runner_lib",
    srcs = [
        "tfRunner.cc",
    ],
    hdrs = [
        "tfRunner.h",
    ],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/core:android_tensorflow_lib",
    ],
)

cc_binary(
  name = "train_and_test_cifar",
  srcs = [
      "train_and_test.cc",
      "util.h",
      "tfRunner.h",
      "tfRunner.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
      "//tensorflow/cc:cc_ops",
      "//tensorflow/core:android_tensorflow_lib",
      ":cifar10Reader",
      ":tf_runner_lib",
      "//external:libTFlogger",
  ],
)


cc_binary(
  name = "train_cifar",
  srcs = [
      "train_cifar10.cc",
      "util.h",
      "tfRunner.h",
      "tfRunner.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
      "//tensorflow/cc:cc_ops",
      "//tensorflow/core:android_tensorflow_lib",
      ":cifar10Reader",
      ":tf_runner_lib",
  ],
)
