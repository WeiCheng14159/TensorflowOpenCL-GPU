# Description:
#   OpenCL hello world program for android

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

cc_library(
    name = "android_opencl_libs",
    srcs = [
        "//external:libopencl_64",
    ],
)

cc_binary(
    name = "opencl-matmul",
    srcs = [
        "opencl-matmul.cc",
    ],
    copts = tf_copts() + [
        "-ffunction-sections",
        "-fdata-sections",
        "-fPIE",
        "-pie",
        "-fexceptions",
    ],
    linkopts = [
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
    ],
    deps = [
        ":android_opencl_libs",
        "//external:libopencl_header",
        "//tensorflow/core:android_tensorflow_lib",
    ],
)
