# Description:
#   TensorFlow C++ training example for MNIST dataset

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_cc_binary",
  )

cc_binary(
    name = "train_mnist",
    srcs = [
        "main.cc",
    ],
    copts = select({
      "//tensorflow:android": [
          "-std=c++11",
          "-DSELECTIVE_REGISTRATION",
          "-DSUPPORT_SELECTIVE_REGISTRATION"
      ],
      "//conditions:default": [],
    }),
    linkopts = select({
        "//tensorflow:android": [
            "-pie",
            "-landroid",
            "-ljnigraphics",
            "-llog",
            "-lm",
            "-z defs",
            "-s",
            "-Wl,--exclude-libs,ALL",
        ],
        "//conditions:default": ["-lm"],
    }),
    deps = select({
      "//tensorflow:android": [
          # cc:cc_ops is used to include image ops (for label_image)
          # Jpg, gif, and png related code won't be included
          "//tensorflow/cc:cc_ops",
          "//tensorflow/core:android_tensorflow_lib",
          # cc:android_tensorflow_image_op is for including jpeg/gif/png
          # decoder to enable real-image evaluation on Android
          "//external:libmnist",
      ],
      "//conditions:default": [
          "//tensorflow/cc:cc_ops",
          "//tensorflow/core:core_cpu",
          "//tensorflow/core:framework",
          "//tensorflow/core:framework_internal",
          "//tensorflow/core:lib",
          "//tensorflow/core:protos_all_cc",
          "//tensorflow/core:tensorflow",
          "//external:libmnist",
      ],
    }),
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
            "bin/**",
            "gen/**",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
