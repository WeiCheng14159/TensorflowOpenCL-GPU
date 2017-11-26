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


# Build the native .so.
# bazel build //tensorflow/contrib/android:libtensorflow_inference.so \
#   --crosstool_top=//external:android/crosstool \
#   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
#   --cpu=armeabi-v7a

cc_binary(
    name = "cpu-mlp-trainer",
    srcs = ["cpu-mlp-trainer.cc"],
    copts = tf_copts() + [
        "-ffunction-sections",
        "-fdata-sections",
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
        "-Wl,--version-script",
        "//tensorflow/c:version_script.lds",
    ],
    linkstatic = 1,
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        "//tensorflow/c:c_api",
        "//tensorflow/core:android_tensorflow_lib",
        "//tensorflow/c:version_script.lds",
    ],
)
