#include "CL/cl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

using namespace std;

///
//  Attempt to create the program object from a cached binary.  Note that
//  on first run this will fail because the binary has not yet been created.
//
cl_program CreateProgramFromBinary(cl_context context, cl_device_id device, const char* fileName)
{
    FILE *fp = fopen(fileName, "rb");
    if (fp == NULL)
    {
        return NULL;
    }

    // Determine the size of the binary
    size_t binarySize;
    fseek(fp, 0, SEEK_END);
    binarySize = ftell(fp);
    rewind(fp);

    unsigned char *programBinary = new unsigned char[binarySize];
    fread(programBinary, 1, binarySize, fp);
    fclose(fp);

    cl_int errNum = 0;
    cl_program program;
    cl_int binaryStatus;

    program = clCreateProgramWithBinary(context,
                                        1,
                                        &device,
                                        &binarySize,
                                        (const unsigned char**)&programBinary,
                                        &binaryStatus,
                                        &errNum);
    delete [] programBinary;
    if (errNum != CL_SUCCESS)
    {
        std::cerr << "Error loading program binary." << std::endl;
        return NULL;
    }

    if (binaryStatus != CL_SUCCESS)
    {
        std::cerr << "Invalid binary for device" << std::endl;
        return NULL;
    }

    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);

        std::cerr << "Error in program: " << std::endl;
        std::cerr << buildLog << std::endl;
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

int read_file(char **output, size_t *size, const char *name) {
  FILE *fp = fopen(name, "rb");
  if (!fp) {
    return -1;
  }

  fseek(fp, 0, SEEK_END);
  *size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  *output = (char *)malloc(*size);
  if (!*output) {
    fclose(fp);
    return -1;
  }

  fread(*output, *size, 1, fp);
  fclose(fp);
  return 0;
}

int write_file(const char *name, const unsigned char *content, size_t size) {
  FILE *fp = fopen(name, "wb+");
  if (!fp) {
    return -1;
  }
  fwrite(content, size, 1, fp);
  fclose(fp);
  return 0;
}

cl_int get_platform_list(cl_platform_id **platforms_out,
                         cl_uint *num_platforms_out) {
  cl_int err;

  // Read the number of platforms
  cl_uint num_platforms;
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  if (err != CL_SUCCESS) {
    return err;
  }
  if (num_platforms == 0) {
    return CL_INVALID_VALUE;
  }

  // Allocate the array of cl_platform_id
  cl_platform_id *platforms =
    (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
  if (!platforms) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Get the result
  err = clGetPlatformIDs(num_platforms, platforms, NULL);
  if (err != CL_SUCCESS) {
    free(platforms);
    return err;
  }

  *platforms_out = platforms;
  *num_platforms_out = num_platforms;
  return CL_SUCCESS;
}

void free_platform_list(cl_platform_id *platforms, cl_uint num_platforms) {
  free(platforms);
}

char *get_platform_info(cl_platform_id platform, cl_platform_info param) {
  cl_int err;

  // Read the size of the buffer for platform name
  size_t buf_size;
  err = clGetPlatformInfo(platform, param, 0, NULL, &buf_size);
  if (err != CL_SUCCESS) {
    return NULL;
  }
  if (buf_size == 0) {
    return NULL;
  }

  // Allocate the buffer for platform name
  char *buf = (char *)malloc(buf_size);
  if (!buf) {
    return NULL;
  }

  // Read the platform name
  err = clGetPlatformInfo(platform, param, buf_size, buf, NULL);
  if (err != CL_SUCCESS) {
    free(buf);
    return NULL;
  }

  return buf;
}

cl_int get_device_list(cl_device_id **devices_out, cl_uint *num_devices_out,
                       cl_platform_id platform) {
  cl_int err;

  // Read the number of devices of the given platform
  cl_uint num_devices;
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL,
                       &num_devices);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Allocate the array of cl_device_id
  cl_device_id *devices =
    (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
  if (!devices) {
    return CL_OUT_OF_HOST_MEMORY;
  }

  // Read the result
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                       devices, NULL);
  if (err != CL_SUCCESS) {
    free(devices);
    return err;
  }

  *devices_out = devices;
  *num_devices_out = num_devices;
  return CL_SUCCESS;
}

void free_device_list(cl_device_id *devices, cl_uint num_devices) {
  cl_uint i;
  for (i = 0; i < num_devices; ++i) {
    clReleaseDevice(devices[i]);
  }
  free(devices);
}

cl_int write_binaries(cl_program program, unsigned num_devices,
                      cl_uint platform_idx) {
  unsigned i;
  cl_int err = CL_SUCCESS;
  size_t *binaries_size = NULL;
  unsigned char **binaries_ptr = NULL;

  // Read the binaries size
  size_t binaries_size_alloc_size = sizeof(size_t) * num_devices;
  binaries_size = (size_t *)malloc(binaries_size_alloc_size);
  if (!binaries_size) {
    err = CL_OUT_OF_HOST_MEMORY;
    return err;
  }

  err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                         binaries_size_alloc_size, binaries_size, NULL);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Read the binaries
  size_t binaries_ptr_alloc_size = sizeof(unsigned char *) * num_devices;
  binaries_ptr = (unsigned char **)malloc(binaries_ptr_alloc_size);
  if (!binaries_ptr) {
    err = CL_OUT_OF_HOST_MEMORY;
    return err;
  }
  memset(binaries_ptr, 0, binaries_ptr_alloc_size);
  for (i = 0; i < num_devices; ++i) {
    binaries_ptr[i] = (unsigned char *)malloc(binaries_size[i]);
    if (!binaries_ptr[i]) {
      err = CL_OUT_OF_HOST_MEMORY;
      return err;
    }
  }

  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binaries_ptr_alloc_size,
                         binaries_ptr, NULL);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Write the binaries to file
  for (i = 0; i < num_devices; ++i) {
    // Create output file name
    char filename[128];
    snprintf(filename, sizeof(filename), "cl-matmul_%u-%u.bin",
             (unsigned)platform_idx, (unsigned)i);

    // Write the binary to the output file
    write_file(filename, binaries_ptr[i], binaries_size[i]);
  }

  return err;
}

cl_int compile_program(cl_uint *num_devices_out, const char *src,
                       size_t src_size, cl_platform_id platform,
                       cl_uint platform_idx) {
  cl_int err = CL_SUCCESS;

  // Get the device list
  cl_device_id* devices = NULL;
  cl_uint num_devices = 0;
  get_device_list(&devices, &num_devices, platform);
  *num_devices_out = num_devices;

  // Create context
  cl_context_properties ctx_properties[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0
  };

  cl_context ctx = clCreateContext(ctx_properties, num_devices, devices, NULL,
                                   NULL, &err);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Create program
  cl_program program = clCreateProgramWithSource(ctx, 1, &src, &src_size, &err);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Compile program
  err = clBuildProgram(program, num_devices, devices, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    return err;
  }

  // Write the binaries
  write_binaries(program, num_devices, platform_idx);

  return err;
}

void compile_all(const char *src, size_t src_size) {
  cl_uint i;

  // Get the platform list
  cl_platform_id *platforms = NULL;
  cl_uint num_platforms = 0;
  if (get_platform_list(&platforms, &num_platforms) != CL_SUCCESS) {
    return;
  }

  // For each platform compile binaries for each devices
  for (i = 0; i < num_platforms; ++i) {
    // Compile for each devices
    cl_uint num_devices = 0;
    cl_int err = compile_program(&num_devices, src, src_size, platforms[i], i);

    // Print the result
    char *platform_name = get_platform_info(platforms[i], CL_PLATFORM_NAME);
    printf("PLATFORM [%s]  -->  %s (%u)\n",
           (platform_name ? platform_name : ""),
           ((err == CL_SUCCESS) ? "SUCCESS" : "FAILURE"),
           (unsigned)num_devices);
    fflush(stdout);
    free(platform_name);
  }

  // Free the platform list
  free_platform_list(platforms, num_platforms);
}

int main(int argc, char **argv) {
  // Check the command line option
  if (argc < 2) {
    cerr << "USAGE: cl-compile [SOURCE]\n";
    exit(EXIT_FAILURE);
  }

  const char *filename = argv[1];

  // Read the source file
  char *src = NULL;
  size_t src_size = 0;
  if (read_file(&src, &src_size, filename) != 0) {
    cerr << "ERROR: Failed to read:" << filename << endl; return -1;
  }

  // Compile binaries for each platforms and devices
  compile_all(src, src_size);

  // Free the source file
  free(src);

  // Get the platform list
  cl_int err = CL_SUCCESS;
  cl_platform_id *platforms = NULL;
  cl_uint num_platforms = 0;
  if (get_platform_list(&platforms, &num_platforms) != CL_SUCCESS) {
    cerr << "ERROR: Failed to get_platform_list" << endl; return -1;
  }

  // Get the device list
  cl_device_id* devices = NULL;
  cl_uint num_devices = 0;
  get_device_list(&devices, &num_devices, platforms[0]);

  // Create a new context
  cl_context_properties ctx_properties[] = {
    CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0
  };

  cl_context ctx = clCreateContext(ctx_properties, num_devices, devices, NULL,
                                   NULL, &err);
  if (err != CL_SUCCESS) {
    cerr << "fail to create contenxt" << endl; return -1;
  }

  // Create a new program
  cl_program prog_from_bin;
  const char * src_fn = "cl-matmul_0-0.bin";

  // Load the kernel binary
  prog_from_bin = CreateProgramFromBinary(ctx, *devices, src_fn);
  if ( !prog_from_bin ){
    cerr << "Fail to create program" << endl; return -1;
  }else{
    cout << "Program created from binary file " << src_fn << endl;
  }

  return 0;
}
