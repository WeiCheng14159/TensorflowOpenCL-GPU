package(default_visibility = ["//visibility:public"])

load(
    "//tensorflow:tensorflow.bzl",
    "tf_copts",
)

android_copts = tf_copts() + [
                    "-ffunction-sections",
                    "-fdata-sections",
                    "-fPIE",
                    "-pie", 
                    "-fexceptions", 
                ]

android_linkopts = [
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

# access_sycl_cl_types
cc_binary(
    name = "access_sycl_cl_types",
    srcs = [   
        "src/access_sycl_cl_types.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# anatomy_sycl_app_parallel_for
cc_binary(
    name = "anatomy_sycl_app_parallel_for",
    srcs = [   
        "src/anatomy_sycl_app_parallel_for.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# anatomy_sycl_app_single_task
cc_binary(
    name = "anatomy_sycl_app_single_task",
    srcs = [   
        "src/anatomy_sycl_app_single_task.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# example_sycl_app
cc_binary(
    name = "example_sycl_app",
    srcs = [   
        "src/example_sycl_app.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# functors_nd_range_kernels
cc_binary(
    name = "functors_nd_range_kernels",
    srcs = [   
        "src/functors_nd_range_kernels.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# naive_square_matrix_rotation
cc_binary(
    name = "naive_square_matrix_rotation",
    srcs = [   
        "src/naive_square_matrix_rotation.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# random_number_generation
cc_binary(
    name = "random_number_generation",
    srcs = [   
        "src/random_number_generation.cpp",
        "common.h",
    ],
     copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# reduction_sum
cc_binary(
    name = "reduction_sum",
    srcs = [   
        "src/reduction_sum.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# reduction_sum_local
cc_binary(
    name = "reduction_sum_local",
    srcs = [   
        "src/reduction_sum_local.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# simple_vector_addition
cc_binary(
    name = "simple_vector_addition",
    srcs = [   
        "src/simple_vector_addition.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# vectors_in_kernel
cc_binary(
    name = "vectors_in_kernel",
    srcs = [   
        "src/vectors_in_kernel.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
# work_efficient_prefix_sum
cc_binary(
    name = "work_efficient_prefix_sum",
    srcs = [   
        "src/work_efficient_prefix_sum.cpp",
        "common.h",
    ],
    copts = android_copts,
    linkopts = android_linkopts,
    linkstatic = 1,
    deps = [   
        "//external:libsycl",          
    ],
)
