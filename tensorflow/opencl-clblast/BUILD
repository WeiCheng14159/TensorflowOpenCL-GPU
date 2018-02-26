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

cc_binary(
  name = "opencl-sgemm",
  srcs = [
    "sgemm.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
    "//external:openclBLAS_libs",
    "//external:android_opencl_libs",
  ],
)

cc_binary(
  name = "opencl-sgemm-tuned",
  srcs = [
    "sgemm.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
    "//external:openclBLAS_libs_for_Adreno540",
    "//external:android_opencl_libs",
  ],
)

cc_binary(
  name = "opencl-hgemm",
  srcs = [
    "hgemm.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
    "//external:openclBLAS_libs",
    "//external:android_opencl_libs",
  ],
)

cc_binary(
  name = "opencl-hgemm-tuned",
  srcs = [
    "hgemm.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
    "//external:openclBLAS_libs_for_Adreno540",
    "//external:android_opencl_libs",
  ],
)

cc_binary(
  name = "opencl-sgemm_batched",
  srcs = [
    "sgemm_batched.cc",
  ],
  copts = ANDROID_C_OPTS,
  linkopts = ANDROID_LINK_OPTS,
  deps = [
    "//external:openclBLAS_libs",
    "//external:android_opencl_libs",
  ],
)
