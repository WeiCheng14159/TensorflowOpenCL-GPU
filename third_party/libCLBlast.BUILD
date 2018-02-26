# An OpenCL BLAS (Basic Linear Algebra Library)

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

filegroup(
    name = "clblast-srcs",
    srcs = glob([
      "src/**/*.cpp",
    ], exclude = [
      "src/clblast_cuda.cpp", 
    ] ),
)

cc_library(
    name = "clblast_libs",
    hdrs = [
      "include/clblast_c.h",
      "include/clblast_half.h",
      "include/clblast.h",
    ],
    includes = [
      "include",
      "src",
    ],
    srcs = [
      ":clblast-srcs",
    ],
    deps = [
      "//external:android_opencl_libs",
    ],
)
