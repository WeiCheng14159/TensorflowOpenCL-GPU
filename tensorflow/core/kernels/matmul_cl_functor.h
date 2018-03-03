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
  template<class T> class clEngine {
    public:

      clEngine(){}

      // clEngine initializaiotn function
      cl_int init(
        typename functor::MatMulTypes<T>::in_type in0,
        typename functor::MatMulTypes<T>::in_type in1)
      {
        // Matrix dimension init
        M = in0.dimension(0);
        K = in0.dimension(1);
        N = in1.dimension(1);

        // Matrix size init
        in0_size = sizeof(T) * M * K;
        in1_size = sizeof(T) * K * N;
        out_size = sizeof(T) * M * N;

        // OpenCL error code init
        err = CL_SUCCESS;

        // Query platforms
        err = clGetPlatformIDs(1, &platform, NULL);
        if( err != CL_SUCCESS )
          return err;

        // Query devices
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
        if( err != CL_SUCCESS )
          return err;

        // Create context
        clCtx = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

        // Create command clQueue
        clQueue = clCreateCommandQueue(clCtx, device, 0, NULL);

        return CL_SUCCESS;
      }

      // Release all OpenCL related resourcse
      cl_int release(){

        // Free OpenCL memory objects
        if( clReleaseMemObject(a) != CL_SUCCESS ||
            clReleaseMemObject(b) != CL_SUCCESS ||
            clReleaseMemObject(c) != CL_SUCCESS )
        {
          return CL_INVALID_MEM_OBJECT;
        }

        // Free OpenCL command queue
        err = CL_SUCCESS;
        err = clReleaseCommandQueue(clQueue);
        if( err != CL_SUCCESS )
          return err;

        // Free OpenCL context
        err = CL_SUCCESS;
        err = clReleaseContext(clCtx);
        if( err != CL_SUCCESS )
          return err;

        // Free OpenCL events
        if( clReleaseEvent(kernel_event)          != CL_SUCCESS ||
            clReleaseEvent(writeBuffer_events[0]) != CL_SUCCESS ||
            clReleaseEvent(writeBuffer_events[1]) != CL_SUCCESS ||
            clReleaseEvent(writeBuffer_events[2]) != CL_SUCCESS )
        {
          return CL_INVALID_EVENT;
        }

        // Return CL_SUCCESS if all resources are released successfully
        return CL_SUCCESS;
      }

      // Load computed results back to memroy
      cl_int mem_load(typename functor::MatMulTypes<T>::out_type out){

        // Init cl err code
        err = CL_SUCCESS;

        // Read results
        err = clEnqueueReadBuffer(clQueue, c, CL_TRUE, 0, out_size, out.data(), 0, NULL, NULL);
        if( err != CL_SUCCESS ){ return err; }

        // Release OpenCL resources
        err = release();
        if( err != CL_SUCCESS ){ return err; }

        // Return if the results are loaded to memory & OpenCL resources are released
        return CL_SUCCESS;
      }

      // OpenCL memeory object init
      cl_int mem_init(
        typename functor::MatMulTypes<T>::in_type in0,
        typename functor::MatMulTypes<T>::in_type in1,
        typename functor::MatMulTypes<T>::out_type out)
      {

        // Init cl err code
        err = CL_SUCCESS;

        // Allocate memory buffers
        a = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, in0_size, NULL, NULL);
        b = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, in1_size, NULL, NULL);
        c = clCreateBuffer(clCtx, CL_MEM_READ_WRITE, out_size, NULL, NULL);

        // Enqueue write buffer commands (acynchronous write)
        err = clEnqueueWriteBuffer(clQueue, a, CL_FALSE, 0, in0_size, in0.data(),
                                   0, NULL, &writeBuffer_events[0]);
        if( err != CL_SUCCESS ){ return err; }

        err = clEnqueueWriteBuffer(clQueue, b, CL_FALSE, 0, in1_size, in1.data(),
                                   0, NULL, &writeBuffer_events[1]);
        if( err != CL_SUCCESS ){ return err; }
          // Write to buffer c to cverwrite previous results
        err = clEnqueueWriteBuffer(clQueue, c, CL_FALSE, 0, out_size, out.data(),
                                   0, NULL, &writeBuffer_events[2]);
        if( err != CL_SUCCESS ){ return err; }

        // Wait for completion
        clWaitForEvents(3, writeBuffer_events);
        return CL_SUCCESS;
      }

      // Compile & Compute the results
      cl_int compileAndComputer(
        const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair )
      {
        // Whether Matrix A, B should be transposed
        auto MatATranspose = ( dim_pair[0].first == 0 ) ?
                              CLBlastTransposeYes : CLBlastTransposeNo;
        auto MatBTranspose = ( dim_pair[0].second == 0 ) ?
                              CLBlastTransposeNo : CLBlastTransposeYes;

        // Leading dimension of the input A matrix. This value must be greater than 0.
        size_t a_ld;

        // Leading dimension of the input B matrix. This value must be greater than 0.
        size_t b_ld;

        // When transpose_a == Transpose::kNo, then a_ld must be at least m,
        // otherwise a_ld must be at least k.
        if( MatATranspose == CLBlastTransposeYes ){
          a_ld = M;
        }else{
          a_ld = K;
        }

        // When transpose_b == Transpose::kNo, then b_ld must be at least k,
        // otherwise b_ld must be at least n.
        if( MatBTranspose == CLBlastTransposeYes ){
          b_ld = K;
        }else{
          b_ld = N;
        }

        // The value of c_ld must be at least m.
        const size_t c_ld = N;

        // Performs the matrix product C = alpha * A * B + beta * C
        const T alpha = 1.0f;
        const T beta = 0.0f;

        // Call the SGEMM routine.
        CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                                MatATranspose, MatBTranspose,
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
          return CL_FALSE;
        }

        clWaitForEvents(1, &kernel_event);
        return CL_SUCCESS;
      }

      // Print debug info
      void debug( bool print=true ){
        if( print ){
          LOG(INFO) << "Dealing with datatype of size " << sizeof(T);
          LOG(INFO) << "in0 = [" << M << "," << K  << "]";
          LOG(INFO) << "in1 = [" << K << "," << N  << "]";
          LOG(INFO) << "out = [" << M << "," << N  << "]";
        }
      }

      // Print input output matrices
      void printMatrix(
        typename functor::MatMulTypes<T>::in_type in0,
        typename functor::MatMulTypes<T>::in_type in1,
        typename functor::MatMulTypes<T>::out_type out)
      {
        LOG(INFO) << "MatMul Matrix details";
        LOG(INFO) << std::endl << in0;
        LOG(INFO) << std::endl << in1;
        LOG(INFO) << std::endl << out;
      }

    private:

      // Default matrix dimension
      size_t M = 0;
      size_t N = 0;
      size_t K = 0;

      // Default matrix size
      size_t in0_size = 0;
      size_t in1_size = 0;
      size_t out_size = 0;

      // OpenCL host side object
      cl_platform_id platform;
      cl_device_id device;
      cl_context clCtx;
      cl_command_queue clQueue;
      cl_int err = CL_SUCCESS;

      // OpenCL memeory object
      cl_mem a;
      cl_mem b;
      cl_mem c;

      // OpenCL events
      cl_event kernel_event;
      cl_event writeBuffer_events[3];

  };  // class clEngine

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
        const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair)
      {

      // Init clEngine with type float
      clEngine<float> c = clEngine<float>();

      // Init cl status
      cl_int status = CL_SUCCESS;

      // OpenCL host & device side initializaiotn
      status = c.init(in0, in1);
      if( status != CL_SUCCESS ){
        LOG(ERROR) << "CL init fail with code " << status;
      }

      // debug info
      // c.debug(true);

      // OpenCL memeory obj init & memory copy
      status = c.mem_init(in0, in1, out);
      if( status != CL_SUCCESS ){
        LOG(ERROR) << "CL mem init fail with code " << status;
      }

      // GEMM computation
      status = c.compileAndComputer(dim_pair);
      if( status != CL_SUCCESS ){
        LOG(ERROR) << "CL compute fail with code " << status;
      }

      // OpenCL memory load
      status = c.mem_load(out);
      if( status != CL_SUCCESS ){
        LOG(ERROR) << "CL memeory load fail with code " << status;
      }

      // Results
      // c.printMatrix(in0, in1, out);

    }
  };

}  // end namespace functor
}  // end namespace tensorflow

#endif  // MATMUL_CL_FUNCTOR_H_
#endif  // TEST_CL
