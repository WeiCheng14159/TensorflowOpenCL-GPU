package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
    "if_android_arm64",
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
    name = "cpu-mlp-trainer",
    srcs = ["cpu-mlp-trainer.cc"],
    copts = ANDROID_C_OPTS,
    linkopts = ANDROID_LINK_OPTS,
    deps = [
        "//external:android_opencl_libs",
        "//tensorflow/core:android_tensorflow_lib",
    ],
)
