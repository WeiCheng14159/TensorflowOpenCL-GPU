# Description:
#   OpenCL hello world program for android

package(default_visibility = ["//visibility:public"])

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
  name = "timerLib",
  srcs = [
    "Timer.cc",
  ],
  hdrs = [
    "Timer.h",
  ],
)

cc_library(
  name = "clMemTester",
  srcs = [
    "clMemTester.cc",
  ],
  hdrs = [
    "clMemTester.h",
  ],
  deps = [
    "//external:android_opencl_libs",
    ":timerLib",
  ],
)

cc_binary(
  name = "opencl-host_to_device",
  srcs = [
    "host_to_device.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
    ":clMemTester",
  ],
  defines = [
    "NUM_OF_TESTS=1",
    "MAX_ALLOCATED=700",
  ],
)

cc_binary(
  name = "opencl-device_to_host",
  srcs = [
    "host_to_device.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
    ":clMemTester",
  ],
  defines = [
    "NUM_OF_TESTS=1",
    "MAX_ALLOCATED=700",
  ],
)

cc_binary(
  name = "opencl-device_to_device",
  srcs = [
    "device_to_device.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
    ":clMemTester",
  ],
  defines = [
    "NUM_OF_TESTS=1",
    "MAX_ALLOCATED=700",
  ],
)
