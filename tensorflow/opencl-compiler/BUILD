# Description:
#   OpenCL kernel builder & saver

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
    tags = [
        "manual",
        "notap",
    ],
)

cc_binary(
    name = "opencl-compiler",
    srcs = [
        "opencl-compiler.c",
    ],
    copts = [
        "-fPIE",
        "-pie",
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
    ],
)
