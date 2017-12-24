# Description: OpenCL driver for Xiaomi 6 phone (Snadragon 835 CPU + Adreno 540 GPU)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

filegroup(
    name = "32-bit-version",
    srcs = [
        "libOpenCL_32.so",
    ],  
)

filegroup(
    name = "64-bit-version",
    srcs = [
        "libOpenCL_64.so",
    ],  
)

cc_library(
    name = "clheader",
    hdrs = glob([
        "CL/**",
        "*.hpp",
    ]),
)
