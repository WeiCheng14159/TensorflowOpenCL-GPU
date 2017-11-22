# Description: OpenCL 1.2 implementation of SYCL 1.2 

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

cc_library(
    name = "adreno540_CL_driver",
    srcs = [        
        "libs/libOpenCL_64.so",
    ],
    linkstatic = 1,
)

cc_library(
    name = "sycl-lib",
    srcs = glob([
        "include/**",
        "source/**",
    ]),
    includes = ["include"],
    deps = [
        ":adreno540_CL_driver", 
    ],
    linkstatic = 1, 
    visibility = ["//visibility:public"],
)

