# Description:
#   OpenCL kernel builder & saver

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

cc_binary(
    name = "opencl-compiler",
    srcs = [
        "opencl-compiler.cc",
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
        "//external:android_opencl_libs",
    ],
)
