#ifdef TEST_CL
  #warning "Complied with TEST_CL flag, TF OpenCL matrix multiplaction will be used!"
#ifndef MATMUL_CL_FUNCTOR_H_
#define MATMUL_CL_FUNCTOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the CLBlast library (C interface)
#include "clblast_c.h"

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

        // Create context
        cl_context clCtx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

        // Create command clQueue
        cl_command_queue clQueue = clCreateCommandQueue(clCtx, device, 0, NULL);

        // Allocate memory buffers
        cl_mem a = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, in0_size, NULL, NULL);
        cl_mem b = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, in1_size, NULL, NULL);
        cl_mem c = clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, out_size, NULL, NULL);

        // Enqueue write buffer commands
        cl_event writeBuffer_events[2];
        err = clEnqueueWriteBuffer(clQueue, a, CL_FALSE, 0, in0_size,
                                  in0.data(), 0, NULL, &writeBuffer_events[0]);
        err = clEnqueueWriteBuffer(clQueue, b, CL_FALSE, 0, in1_size,
                                  in1.data(), 0, NULL, &writeBuffer_events[1]);

        // Wait for completion
        clWaitForEvents(2, writeBuffer_events);

        const T alpha = 1.0f;
        const T beta = 0.0f;
        const size_t a_ld = M;
        const size_t b_ld = K;
        const size_t c_ld = M;

        // Create kernel_event
        cl_event kernel_event = NULL;

        // Call the SGEMM routine.
        CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                                CLBlastTransposeNo, CLBlastTransposeNo,
                                                M, N, K,
                                                alpha,
                                                a, 0, a_ld,
                                                b, 0, b_ld,
                                                beta,
                                                c, 0, c_ld,
                                                &clQueue, &kernel_event);

        // Wait for completion
        if (status != CLBlastSuccess){
          LOG(ERROR) << "[CLBlast] Fail with code " << status;
          return;
        }

        // Read results
        clEnqueueReadBuffer(clQueue, c, CL_TRUE, 0, out_size, out.data(), 1, &kernel_event, NULL);

        // Free the OpenCL memory objects
        clReleaseMemObject(a);
        clReleaseMemObject(b);
        clReleaseMemObject(c);

        // Clean-up OpenCL
        clReleaseCommandQueue(clQueue);
        clReleaseContext(clCtx);
        clReleaseEvent(kernel_event);
        clReleaseEvent(writeBuffer_events[0]);
        clReleaseEvent(writeBuffer_events[1]);

        // LOG(INFO) << "in0 = [" << M << "," << K  << "]";
        // LOG(INFO) << endl << in0;
        // LOG(INFO) << "in1 = [" << K << "," << N  << "]";
        // LOG(INFO) << endl << in1;
        // LOG(INFO) << "out = [" << M << "," << N  << "]";
        // LOG(INFO) << endl << out;
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
