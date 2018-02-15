#ifdef TEST_CL
  #warning "Complied with TEST_CL flag, TF OpenCL matrix multiplaction will be used!"
#ifndef MATMUL_CL_FUNCTOR_H_
#define MATMUL_CL_FUNCTOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"

#include "CL/cl.h"

using namespace std;

namespace tensorflow {
  typedef Eigen::ThreadPoolDevice CPUDevice;

  // The OpenCL matrix multiplication computer, computing datatype T
  template<class T> class ClComputer {
    public:
      ClComputer(typename functor::MatMulTypes<T>::in_type in0,
                 typename functor::MatMulTypes<T>::in_type in1,
                 typename functor::MatMulTypes<T>::out_type out)
      {
        const size_t M = in0.dimension(0);
        const size_t K = in0.dimension(1);
        const size_t N = in1.dimension(1);
        const size_t in0_size = sizeof(T) * M * K;
        const size_t in1_size = sizeof(T) * K * N;
        const size_t out_size = sizeof(T) * M * N;

        cl_int err = CL_SUCCESS;

        // Query platforms and devices
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, NULL);

        cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);

        const cl_context_properties prop[] = {
          CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
          0
        };

        // Create context
        cl_context clCtx = clCreateContext(prop, 1, &device, NULL, NULL, &err);

        // Create program
        unsigned char* clKernelBinaryFile = NULL;
        size_t clKernelBinSize = 0;
        // Read compiled binary
        read_file(&clKernelBinaryFile, &clKernelBinSize, clKernelBinName.c_str() );

        cl_program clProgram =
          clCreateProgramWithBinary(clCtx, 1, &device, &clKernelBinSize,
                                    (const unsigned char **)&clKernelBinaryFile,
                                    NULL, &err);

        err = clBuildProgram(clProgram, 1, &device, NULL, NULL, NULL);

        // Allocate memory buffers
        cl_mem a = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, in0_size, NULL, &err);
        cl_mem b = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, in1_size, NULL, &err);
        cl_mem c = clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, out_size, NULL, &err);

        // Create command clQueue
        cl_command_queue clQueue = clCreateCommandQueue(clCtx, device, 0, NULL);

        // Enqueue write buffer commands
        cl_event writeBuffer_events[2];
        err = clEnqueueWriteBuffer(clQueue, a, CL_FALSE, 0, in0_size,
                                  in0.data(), 0, NULL, &writeBuffer_events[0]);
        err = clEnqueueWriteBuffer(clQueue, b, CL_FALSE, 0, in1_size,
                                  in1.data(), 0, NULL, &writeBuffer_events[1]);

        // Enqueue the kernel execution command
        cl_kernel clKernel = clCreateKernel(clProgram, clKernelFuncName.c_str() , &err);
        err = clSetKernelArg(clKernel, 0, sizeof(int), &M);
        err = clSetKernelArg(clKernel, 1, sizeof(int), &N);
        err = clSetKernelArg(clKernel, 2, sizeof(int), &K);
        err = clSetKernelArg(clKernel, 3, sizeof(cl_mem), &a);
        err = clSetKernelArg(clKernel, 4, sizeof(cl_mem), &b);
        err = clSetKernelArg(clKernel, 5, sizeof(cl_mem), &c);

        const int TS = 32;
        const size_t local[2] = { TS, TS };
        const size_t global[2] = { M, N };
        cl_event kernel_event;
        err = clEnqueueNDRangeKernel(clQueue, clKernel, 2, NULL,
                                     global, local, 2, writeBuffer_events, &kernel_event);

        // Enqueue the read buffer command
        err = clEnqueueReadBuffer(clQueue, c, CL_TRUE, 0, out_size, out.data(),
                                  1, &kernel_event, NULL);

        // Wait until every commands are finished
        err = clFinish(clQueue);
        if ( err != CL_SUCCESS )
          LOG(ERROR) << "Fail";

        LOG(INFO) << "in0 = [" << M << "," << K  << "]";
        LOG(INFO) << endl << in0;
        LOG(INFO) << "in1 = [" << K << "," << N  << "]";
        LOG(INFO) << endl << in1;
        LOG(INFO) << "out = [" << M << "," << N  << "]";
        LOG(INFO) << endl << out;

        // Free the OpenCL memory objects
        clReleaseMemObject(a);
        clReleaseMemObject(b);
        clReleaseMemObject(c);

        // Clean-up OpenCL
        clReleaseCommandQueue(clQueue);
        clReleaseContext(clCtx);
        clReleaseProgram(clProgram);
        clReleaseKernel(clKernel);

      }

    private:
      std::string clKernelFuncName = "GEMM1";
      std::string clKernelBinName = "matmul.bin";

      // This function reads the compiled cl kernel binary
      int read_file(unsigned char **output, size_t *size, const char *name) {
        FILE* fp = fopen(name, "rb");
        if (!fp) {
          LOG(ERROR) << "Fail to read cl kernel binary " << std::string( name );
        }

        fseek(fp, 0, SEEK_END);
        *size = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        *output = (unsigned char *)malloc(*size);
        if (!*output) {
          fclose(fp);
          return -1;
        }

        fread(*output, *size, 1, fp);
        fclose(fp);
        return 0;
      }
  };  // class ClComputer

namespace functor {

  template <typename Device, typename T>
  struct MatMulCLFunctor {
    // Computes on device "d": out = in0 * in1, where * is matrix
    // multiplication.
    void operator()(
        const Device& d, typename MatMulTypes<T>::out_type out,
        typename MatMulTypes<T>::in_type in0,
        typename MatMulTypes<T>::in_type in1,
        const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair);
  };

  // Partial specialization MatMulFunctor<Device=CPUDevice, T>.
  template <typename T>
  struct MatMulCLFunctor<CPUDevice, T> {
    void operator()(
        const CPUDevice& d, typename MatMulTypes<T>::out_type out,
        typename MatMulTypes<T>::in_type in0,
        typename MatMulTypes<T>::in_type in1,
        const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
      MatMul<CPUDevice>(d, out, in0, in1, dim_pair);
    }
  };

  // Partial specialization MatMulFunctor<Device=CPUDevice, float>
  /*
  Notice that only floating pointing matrix multiplication will be handled by
  OpenCL, other datatype complutation will be handled by Eigen CPU library
  */
  template <>
  struct MatMulCLFunctor<CPUDevice, float> {
    void operator()(
        const CPUDevice& d, typename MatMulTypes<float>::out_type out,
        typename MatMulTypes<float>::in_type in0,
        typename MatMulTypes<float>::in_type in1,
        const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
      ClComputer<float> c = ClComputer<float>( in0, in1, out );
    }
  };

}  // end namespace functor
}  // end namespace tensorflow

#endif  // MATMUL_CL_FUNCTOR_H_
#endif  // TEST_CL
