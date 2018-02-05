# Description:
# OpenCL ops

package(
    default_visibility = ["//tensorflow:internal"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_cuda_tests_tags")
load("//tensorflow:tensorflow.bzl", "if_android", "tf_copts")

exports_files(["LICENSE"])

cc_library(
    name = "android_opencl_libs",
    srcs = [
        "//external:libopencl_64",
    ],
    linkstatic = 1,
)

cc_library(
    name = "zero_out_op_kernel_1.so",
    srcs = ["zero_out_op_kernel_1.cc"],
    deps = [ "//tensorflow/core:framework" ] +
      if_android([
        ":android_opencl_libs",
        "//external:libopencl_header",
        "//tensorflow/core:android_tensorflow_lib",
      ]),
)

cc_binary(
    name = "attr_examples",
    srcs = ["attr_examples.cc"],
    copts = tf_copts(),
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
          "//tensorflow/cc:cc_ops",
          "//tensorflow/core:android_tensorflow_lib",
          ":zero_out_op_kernel_1.so",
      ],
      "//conditions:default": [
          "//tensorflow/core",
          "//tensorflow/core:framework",
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
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
