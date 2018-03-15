# Description: Tensorboad C++ logger
# https://github.com/RustingSword/tensorboard_logger
# https://github.com/supernovaremnant/tensorboard_logger ->  Hacky have to
# change reporitory structure to accomodate Bazel rules

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE.TXT"])

###cc_proto_library###
cc_proto_library(
  name = "event_cc_proto",
  deps = [ ":event_proto" ],
)
cc_proto_library(
  name = "summary_cc_proto",
  deps = [ ":summary_proto" ],
)
cc_proto_library(
  name = "tensor_cc_proto",
  deps = [ ":tensor_proto" ],
)
cc_proto_library(
  name = "tensorShape_cc_proto",
  deps = [ ":tensorShape_proto" ],
)
cc_proto_library(
  name = "types_cc_proto",
  deps = [ ":types_proto" ],
)

###proto_library###
proto_library(
  name = "event_proto",
  srcs = [ "event.proto" ],
  deps = [":summary_proto"],
)
proto_library(
  name = "resHandle_proto",
  srcs = [ "resource_handle.proto" ],
)
proto_library(
  name = "summary_proto",
  srcs = [ "summary.proto" ],
  deps = [":tensor_proto"],
)
proto_library(
  name = "tensor_proto",
  srcs = [
    "tensor.proto",
  ],
  deps = [
    ":resHandle_proto",
    ":tensorShape_proto",
    ":types_proto",
  ],
)
proto_library(
  name = "tensorShape_proto",
  srcs = [ "tensor_shape.proto" ],
)
proto_library(
  name = "types_proto",
  srcs = [ "types.proto" ],
)

###cc_library###
cc_library(
    name = "tensorboard_logger",
    hdrs = glob([
        "include/*.h",
    ]),
    srcs = glob([
        "src/*.cc",
    ]),
    strip_include_prefix = "include",
    deps = [
      ":event_cc_proto",
      ":summary_cc_proto",
      ":tensor_cc_proto",
      ":tensorShape_cc_proto",
      ":types_cc_proto",
    ],
    linkstatic = 1,
)
