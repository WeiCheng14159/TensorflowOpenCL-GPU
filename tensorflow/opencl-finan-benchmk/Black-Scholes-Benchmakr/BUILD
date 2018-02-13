# Description:
#   OpenCL hello world program for android

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "if_android_arm64",
)

cc_library(
    name = "black_scholes_bench_lib",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    deps = [
        "//external:android_opencl_libs",
    ],
)

cc_binary(
    name = "opencl-finan-black-scholes",
    srcs = [
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
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//external:android_opencl_libs",
        ":black_scholes_bench_lib",
    ],
)
