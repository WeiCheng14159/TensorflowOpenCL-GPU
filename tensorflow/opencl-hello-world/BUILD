# Description:
#   OpenCL hello world program for android

package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "if_android_arm64",
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

cc_binary(
    name = "opencl-hello-world",
    srcs = [
        "opencl-hello-world.cc",
    ],
    copts = ANDROID_C_OPTS,
    linkopts = ANDROID_LINK_OPTS,
    deps = [
        "//external:android_opencl_libs",
    ],
)
