# Description: OpenCL driver for Xiaomi 6 phone (Snadragon 835 CPU + Adreno 540 GPU)
# https://github.com/supernovaremnant/Android-OpenCL-Driver

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

filegroup(
    name = "qualcomm_adreno_540_32_bit_lib",
    srcs = [
        "libOpenCL_32.so",
    ],
)

filegroup(
    name = "qualcomm_adreno_540_64_bit_lib",
    srcs = [
        "libOpenCL_64.so",
    ],
)

filegroup(
    name = "arm_mali_t880_64_bit_lib",
    srcs = [
        "libOpenCLIcd.so",
    ],
)

cc_library(
    name = "Qualcomm_Adreno_540_android_opencl_64_bit_lib",
    hdrs = glob([
        "CL/**",
        "*.hpp",
    ]),
    srcs = [
        ":qualcomm_adreno_540_64_bit_lib",
    ],
)

cc_library(
    name = "ARM_Mali_T880_android_opencl_64_bit_lib",
    hdrs = glob([
        "CL/**",
        "*.hpp",
    ]),
    srcs = [
        ":arm_mali_t880_64_bit_lib",
    ],
)
