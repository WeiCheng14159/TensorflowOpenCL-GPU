# Description:
#   LMDB is the Lightning Memory-mapped Database.

licenses(["notice"])  # OpenLDAP Public License

exports_files(["LICENSE"])

cc_library(
    name = "lmdb",
    srcs = [
        "mdb.c",
        "midl.c",
    ],
    hdrs = [
        "lmdb.h",
        "midl.h",
    ],
    copts = select({
      "//conditions:default": ["-w"],
      ":android_arm64": ["-w","-DANDROID"],
    }),
    linkopts = select({
        ":windows": ["-DEFAULTLIB:advapi32.lib"],  # InitializeSecurityDescriptor, SetSecurityDescriptorDacl
        ":windows_msvc": ["-DEFAULTLIB:advapi32.lib"],
        ":android_arm64": [],
        "//conditions:default": ["-lpthread"],
    }),
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

config_setting(
    name = "windows_msvc",
    values = {"cpu": "x64_windows_msvc"},
)

config_setting(
    name = "android_arm64",
    values = {"cpu": "arm64-v8a"},
)
