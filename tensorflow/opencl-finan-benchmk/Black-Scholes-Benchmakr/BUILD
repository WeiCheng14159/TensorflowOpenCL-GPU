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
        ":android_opencl_libs",            
        "//external:libopencl_header",
    ],
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
        ":android_opencl_libs",            
        "//external:libopencl_header",
        ":black_scholes_bench_lib", 
    ],
)
