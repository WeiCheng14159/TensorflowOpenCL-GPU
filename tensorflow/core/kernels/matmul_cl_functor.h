// clMatMulEngine<float>
//     |
//     v
// clQualcommFP32Engine <---- binaryLoaderInterface

// clMatMulEngine<float>
//     |
//     v
// clQualcommFP16Engine <---- binaryLoaderInterface

// clMatMulEngine<float>
//     |
//     v
// clBLASTEngine

#ifndef MATMUL_CL_FUNCTOR_H_
#define MATMUL_CL_FUNCTOR_H_

#include <fstream>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings

// Includes the CLBlast library (C interface)
#include "clblast_c.h"

////////////////////////////////////////////////////////////////////////////////
// OpenCL status checker
#define CL_CHECK(_expr)                                                        \
  {                                                                            \
    cl_int _err = _expr;                                                       \
    if( _err != CL_SUCCESS) {                                                  \
      std::cerr << "OpenCL Error: " << #_expr << " returned " << (int)_err     \
      << std::endl;                                                            \
    }                                                                          \
  }
// OpenCL return type checker
#define CL_CHECK_ERR(_expr)                                                    \
  ({                                                                           \
    cl_int _err = CL_INVALID_VALUE;                                            \
    decltype(_expr) _ret = _expr;                                              \
    if (_err != CL_SUCCESS) {                                                  \
      std::cerr << "OpenCL Error: " << #_expr << " returned " << (int)_err     \
      << std::endl;                                                            \
    }                                                                          \
    _ret;                                                                      \
  })

////////////////////////////////////////////////////////////////////////////////
// float to cl_half conversions
#ifndef INFINITY
  #define INFINITY 1.0/0.0
#endif

#ifndef NAN
  #define NAN 0.0/0.0
#endif

typedef union {
  int32_t i;
  float f;
} FloatConvUnion;

cl_half float_to_cl_half(float value){

  FloatConvUnion u;
  u.f = value;
  cl_half half = (u.i >> 16) & 0x8000; // sign
  cl_half fraction = (u.i >> 12) & 0x007ff; // fraction with extra bit for rounding
  cl_half exponent = (u.i >> 23)  & 0xff; // exponent

  if(exponent < 0x0067) // Return signed zero if zero or value is too small for denormal half
    return half;

  if(exponent > 0x008e){// value was NaN or Inf
    half |= 0x7c00u; // Make into inf
    half |= exponent == 255 && (u.i & 0x007fffffu); // If value was NaN make this into NaN
    return half;
  }

  if(exponent < 0x0071){// Denormal
    fraction |= 0x0800u;

    // rounding
    half |= (fraction >> (0x0072 - exponent)) + ((fraction >> (0x0071 - exponent)) & 1);
    return half;
  }

  half |= ((exponent - 0x0070) << 10) | (fraction >> 1);
  half += fraction & 1;// rounding
  return half;
}

//////////////////////////////////////////////////////////////////////////////////////////
// clSetKernelArg Helper
#define SET_GEMM_TN_KERNEL_ARG(M, K, N, clMemA, clMemB, clMemC, localSize, localMemType, \
  iter)                                                                                  \
  CL_CHECK( clSetKernelArg(clGemmKernel, 0, sizeof(cl_ushort), &M) );                    \
  CL_CHECK( clSetKernelArg(clGemmKernel, 1, sizeof(cl_ushort), &K) );                    \
  CL_CHECK( clSetKernelArg(clGemmKernel, 2, sizeof(cl_ushort), &N) );                    \
  CL_CHECK( clSetKernelArg(clGemmKernel, 3, sizeof(cl_mem), &clMemA) );                  \
  CL_CHECK( clSetKernelArg(clGemmKernel, 4, sizeof(cl_mem), &clMemB) );                  \
  CL_CHECK( clSetKernelArg(clGemmKernel, 5, sizeof(cl_mem), &clMemC) );                  \
  CL_CHECK( clSetKernelArg(clGemmKernel, 6, localSize * sizeof(localMemType), NULL) );   \
  CL_CHECK( clSetKernelArg(clGemmKernel, 7, sizeof(cl_ushort), &iter) );                 \

//////////////////////////////////////////////////////////////////////////////////////////
// clSetKernelArg Helper
#define SET_TRANS_KERNEL_ARG(ROW, COL, clMem, clMem_T, iter)                             \
  CL_CHECK( clSetKernelArg(clTransKernel, 0, sizeof(cl_ushort), &ROW) );                 \
  CL_CHECK( clSetKernelArg(clTransKernel, 1, sizeof(cl_ushort), &COL) );                 \
  CL_CHECK( clSetKernelArg(clTransKernel, 2, sizeof(cl_mem), &clMem) );                  \
  CL_CHECK( clSetKernelArg(clTransKernel, 3, sizeof(cl_mem), &clMem_T) );                \
  CL_CHECK( clSetKernelArg(clTransKernel, 4, sizeof(cl_ushort), &iter) );                \

using namespace std;

namespace tensorflow {
  typedef Eigen::ThreadPoolDevice CPUDevice;

  // clMatMulEngine abstract class (interface), computing datatype T
  template<class T> class clMatMulEngine {
    public:

    // Concrete methods
      // clMatMulEngine initializaiotn function
      cl_int hostInit(
        typename functor::MatMulTypes<T>::in_type in0,
        typename functor::MatMulTypes<T>::in_type in1,
        typename functor::MatMulTypes<T>::out_type out,
        const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair)
      {
        // Matrix dimension init
        RowA = in0.dimension(0);
        ColA = in0.dimension(1);
        RowB = in1.dimension(0);
        ColB = in1.dimension(1);
        RowC = out.dimension(0);
        ColC = out.dimension(1);

        // Matrix size checking
        int matrixSizeLimit = 0xffff; // Maximum value for cl_ushort
        if( RowA > matrixSizeLimit ||
            ColA > matrixSizeLimit ||
            RowB > matrixSizeLimit ||
            ColB > matrixSizeLimit ||
            RowC > matrixSizeLimit ||
            ColC > matrixSizeLimit )
        {
          LOG(ERROR) << "Matrix of Size Larger than " << matrixSizeLimit <<
          " isn't supported";
        }

        // Matrix size init
        a_size = sizeof(T) * RowA * ColA;
        b_size = sizeof(T) * RowB * ColB;
        c_size = sizeof(T) * RowC * ColC;

        // Matrix transpose
        a_traspose = ( dim_pair[0].first == 0 ) ? true : false;
        b_traspose = ( dim_pair[0].second == 1 ) ? true : false;

        // Query platforms
        CL_CHECK( clGetPlatformIDs(1, &platform, NULL) );

        // Query devices
        CL_CHECK( clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &clDevice, NULL) );

        // Create context
        clCtx = CL_CHECK_ERR( clCreateContext(NULL, 1, &clDevice, NULL, NULL, &_err) );

        // Create command clQueue
        clQueue = CL_CHECK_ERR( clCreateCommandQueue(clCtx, clDevice, 0, &_err) );

        return CL_SUCCESS;
      }

      // Print debug info
      void debug( bool print=true ){
        if( print ){
          LOG(INFO) << "Dealing with datatype of size " << sizeof(T);
          LOG(INFO) << "MatrixA = [" << RowA << "," << ColA  << "]";
          LOG(INFO) << "MatrixB = [" << RowB << "," << ColB  << "]";
          LOG(INFO) << "MatrixC = [" << RowC << "," << ColC  << "]";
        }
      }

      // Print input-output matrices
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

    // Virtual methods
      // Release all OpenCL related resourcse
      virtual cl_int clEnd() = 0;

      // Load computed results back to memroy
      virtual cl_int memLoad(typename functor::MatMulTypes<T>::out_type out) = 0;

      // OpenCL memeory object init
      virtual cl_int memInit( typename functor::MatMulTypes<T>::in_type in0,
        typename functor::MatMulTypes<T>::in_type in1) = 0;

    protected:

      // Default matrix dimension
      size_t RowA = 0;
      size_t ColA = 0;
      size_t RowB = 0;
      size_t ColB = 0;
      size_t RowC = 0;
      size_t ColC = 0;

      // Matrix tranpose info
      bool a_traspose;
      bool b_traspose;

      // Default matrix size
      size_t a_size = 0;
      size_t b_size = 0;
      size_t c_size = 0;

      // OpenCL host side object
      cl_platform_id platform;
      cl_device_id clDevice;
      cl_context clCtx;
      cl_command_queue clQueue;

      // Timer
      std::chrono::high_resolution_clock::time_point timer;

      void startTimer(){
        timer = std::chrono::high_resolution_clock::now();
      }

      double read_us(){
        auto elapsed_time = std::chrono::high_resolution_clock::now() - timer;
        return std::chrono::duration<double, std::micro>(elapsed_time).count();
      }

      // Performance calculator
      void getPerformance(){

        std::ofstream ofs ("performance.log", std::ios_base::app);

        double delta_t = read_us() * 1e6; // delta_t in second

        double bandwidth = (a_size+b_size+c_size)/delta_t;
        ofs << bandwidth << ",";

        double flops = (RowA*ColA*RowB*ColB*RowC*ColC)/delta_t;
        ofs << flops << "\n";

        ofs.close();
      }

  };  // class clMatMulEngine

  // binaryLoaderInterface abstract class (interface)
  class binaryLoaderInterface{
    public:

    // Virtual method
      // Compile & Compute the results
      virtual cl_int loadFromBinaryCompute() = 0;

    protected:

    // Concrete methods
      // Read OpenCL binary file from disk
      int read_file(unsigned char **output, size_t *size, const char *name)
      {
        FILE* fp = fopen(name, "rb");
        if (!fp) {
          LOG(ERROR) << "Fail to read cl kernel binary " << std::string( name );
          return -1;
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

      // Show clKernel object info
      void debugOpenclKernel(cl_kernel cl_kernel, cl_device_id cl_device){

        // Kernel info
        size_t wgSize = 0;
        size_t compiledWgSize[3];
        cl_ulong localMemSize = 0;
        size_t perfHint;
        cl_ulong privateMemSize = 0;

        CL_CHECK( clGetKernelWorkGroupInfo(cl_kernel, cl_device,
                    CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wgSize, NULL) );
        CL_CHECK( clGetKernelWorkGroupInfo(cl_kernel, cl_device,
                    CL_KERNEL_COMPILE_WORK_GROUP_SIZE, 3 * sizeof(size_t),
                    &compiledWgSize, NULL) );
        CL_CHECK( clGetKernelWorkGroupInfo(cl_kernel, cl_device,
                    CL_KERNEL_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize,
                    NULL) );
        CL_CHECK( clGetKernelWorkGroupInfo(cl_kernel, cl_device,
                    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t),
                    &perfHint, NULL) );
        CL_CHECK( clGetKernelWorkGroupInfo(cl_kernel, cl_device,
                    CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(cl_ulong), &privateMemSize,
                    NULL) );
        printf("\
          CL_KERNEL_WORK_GROUP_SIZE %zu,\n \
          CL_KERNEL_COMPILE_WORK_GROUP_SIZE [%zu,%zu,%zu],\n \
          CL_KERNEL_LOCAL_MEM_SIZE %ld,\n \
          CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE %zu,\n \
          CL_KERNEL_PRIVATE_MEM_SIZE %ld\n\n",
          wgSize,
          compiledWgSize[0], compiledWgSize[1], compiledWgSize[2],
          localMemSize,
          perfHint,
          privateMemSize);

      }
  };  // class binaryLoaderInterface

  // clQualcommFP32Engine concrete class using Qualcomm GEMM example
  class clQualcommFP32Engine : public binaryLoaderInterface, public clMatMulEngine<float>{
    public:

      cl_int clEnd(){

        // Free OpenCL memory objects
        CL_CHECK( clReleaseMemObject(clBufferA) );
        clReleaseMemObject(clBufferA_T);
        CL_CHECK( clReleaseMemObject(clBufferB) );
        clReleaseMemObject(clBufferB_T);
        CL_CHECK( clReleaseMemObject(clBufferC) );

        // Free OpenCL kernel
        CL_CHECK( clReleaseKernel(clGemmKernel) );
        CL_CHECK( clReleaseKernel(clTransKernel) );

        // Free OpenCL program
        CL_CHECK( clReleaseProgram(clProgram) );

        // Free OpenCL command queue
        CL_CHECK( clReleaseCommandQueue(clQueue) );

        // Free OpenCL context
        CL_CHECK( clReleaseContext(clCtx) );

        // Free OpenCL events
        clReleaseEvent(transKernelEvent[0]);
        clReleaseEvent(transKernelEvent[1]);
        CL_CHECK( clReleaseEvent(gemmKernelEvent) );
        CL_CHECK( clReleaseEvent(mapBufferEvents[0]) );
        CL_CHECK( clReleaseEvent(mapBufferEvents[1]) );
        CL_CHECK( clReleaseEvent(unMapBufferEvents[0]) );
        CL_CHECK( clReleaseEvent(unMapBufferEvents[1]) );

        // Return CL_SUCCESS if all resources are released successfully
        return CL_SUCCESS;
      }

      cl_int memLoad(typename functor::MatMulTypes<float>::out_type out){

        // Use the map function to return clBufferA pointer to the host <= blocking
        clHostPtrC = ( cl_float * ) clEnqueueMapBuffer(clQueue, clBufferC, CL_TRUE,
                          CL_MAP_READ, 0, c_size, 0, NULL, NULL, NULL);

        // Read computed result back to host
        for( auto idx = 0 ; idx < RowC*ColC ; idx++){
          out.data()[idx] = clHostPtrC[idx];
        }

        // Release OpenCL resources
        CL_CHECK( clEnd() );

        // Return if the results are loaded to memory & OpenCL resources are released
        return CL_SUCCESS;
      }

      cl_int memInit(
        typename functor::MatMulTypes<float>::in_type in0,
        typename functor::MatMulTypes<float>::in_type in1)
      {

        // Use zero copy to avoid additional memeory copy
        // Matrix A
        clBufferA = clCreateBuffer(clCtx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      a_size, NULL, NULL);
        // Use the map function to return clBufferA pointer to the host <= non-blocking
        clHostPtrA = ( cl_float * ) clEnqueueMapBuffer(clQueue, clBufferA, CL_FALSE,
                                      CL_MAP_WRITE, 0, a_size, 0, NULL,
                                      &mapBufferEvents[0], NULL);
        // Matrix B
        clBufferB = clCreateBuffer(clCtx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                          b_size, NULL, NULL);
        // Use the map function to return clBufferA pointer to the host <= non-blocking
        clHostPtrB = ( cl_float * ) clEnqueueMapBuffer(clQueue, clBufferB, CL_FALSE,
                                      CL_MAP_WRITE, 0, b_size, 0, NULL,
                                      &mapBufferEvents[1], NULL);

        // Create GPU buffer for transposed matrices only if needed
        if( a_traspose ){
          clBufferA_T = clCreateBuffer(clCtx, CL_MEM_HOST_NO_ACCESS, a_size, NULL, NULL);
        }
        if( !b_traspose ){
          clBufferB_T = clCreateBuffer(clCtx, CL_MEM_HOST_NO_ACCESS, b_size, NULL, NULL);
        }

        // Wait for completion
        CL_CHECK( clWaitForEvents(2, mapBufferEvents) );

        // Host update the buffer using pointer clHostPtrA in host address space
        for( auto idx = 0 ; idx < RowA*ColA ; idx ++){
          clHostPtrA[ idx ] = in0.data()[idx];
        }
        // Host update the buffer using pointer clHostPtrB in host address space
        for( auto idx = 0 ; idx < RowB*ColB ; idx ++){
          clHostPtrB[ idx ] = in1.data()[idx];
        }

        // Unmap the object -> Used in the OpenCL kernel
        CL_CHECK( clEnqueueUnmapMemObject( clQueue, clBufferA, (void*) clHostPtrA,
                    0, NULL, &unMapBufferEvents[0] ) );

        // Unmap the object -> Used in the OpenCL kernel
        CL_CHECK( clEnqueueUnmapMemObject( clQueue, clBufferB, (void*) clHostPtrB,
                    0, NULL, &unMapBufferEvents[1] ) );

        // Matrix C
        clBufferC = clCreateBuffer(clCtx, CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      c_size, NULL, NULL);

        // Wait for completion
        CL_CHECK( clWaitForEvents(2, unMapBufferEvents) );
        return CL_SUCCESS;
      }

      cl_int loadFromBinaryCompute()
      {

        unsigned char* clKernelBinaryFile = NULL;
        size_t clKernelBinSize = 0;
        // Read compiled OpenCL kernel binary file from disk
        read_file(&clKernelBinaryFile, &clKernelBinSize, "matmul.bin" );

        // Create an OpenCL program object from binary
        clProgram = CL_CHECK_ERR( clCreateProgramWithBinary(clCtx, 1, &clDevice,
                                    &clKernelBinSize,
                                    (const unsigned char **)&clKernelBinaryFile,
                                    NULL, &_err) );

        // OpenCL build program
        CL_CHECK( clBuildProgram(clProgram, 1, &clDevice, "-cl-fast-relaxed-math" , NULL, NULL) );

        // Create OpenCL GEMM kernel object
        // clGemmKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatMul_TN_1D_Fp32_Float4" , &_err) );
        // clGemmKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatMul_TN_1D_Fp32_Float8" , &_err) );
        clGemmKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatMul_TN_1D_Fp32_Float16" , &_err) );

        // Create OpenCL Transpose kernel object
        // clTransKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatTrans_1D_Fp32_Float4" , &_err) );
        // clTransKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatTrans_1D_Fp32_Float8" , &_err) );
        clTransKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatTrans_1D_Fp32_Float16" , &_err) );

        cl_ushort gemmKernelIter;
        cl_ushort transKernelIter;

        // Handle Matrices Transpose
        if( a_traspose && b_traspose ){ // Transpose A: yes, Transpose B: yes

          transKernelIter = ColA >> 4;
          gemmKernelIter = RowA >> 4;

          // Transpose A
          SET_TRANS_KERNEL_ARG(RowA, ColA, clBufferA, clBufferA_T, transKernelIter );

          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                      &RowA, NULL, 0, NULL, &transKernelEvent[0]) );

          SET_GEMM_TN_KERNEL_ARG(ColA, RowA, RowB, clBufferA_T, clBufferB,
            clBufferC, ColA, float, gemmKernelIter );

          startTimer();

          const size_t global = ColA;
          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                      &global, NULL, 1, transKernelEvent, &gemmKernelEvent) );

          CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );

        }else if( a_traspose && !b_traspose ){ // Transpose A: yes, Transpose B: no

          transKernelIter = ColA >> 4;
          gemmKernelIter = RowA >> 4;

          // Transpose A
          SET_TRANS_KERNEL_ARG(RowA, ColA, clBufferA, clBufferA_T, transKernelIter );

          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                      &RowA, NULL, 0, NULL, &transKernelEvent[0]) );

          transKernelIter = ColB >> 4;

          // Transpose B
          SET_TRANS_KERNEL_ARG(RowB, ColB, clBufferB, clBufferB_T, transKernelIter );

          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                      &RowB, NULL, 0, NULL, &transKernelEvent[1]) );

          SET_GEMM_TN_KERNEL_ARG(ColA, RowA, ColB, clBufferA_T, clBufferB_T,
            clBufferC, RowA, float, gemmKernelIter );

          startTimer();

          const size_t global = ColA;
          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                      &global, NULL, 2, transKernelEvent, &gemmKernelEvent) );

          CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );

        }else if( !a_traspose && b_traspose ){ // Transpose A: no, Transpose B: yes

          gemmKernelIter = ColA >> 4;

          SET_GEMM_TN_KERNEL_ARG(RowA, ColA, RowB, clBufferA, clBufferB,
            clBufferC, ColA, float, gemmKernelIter );

          startTimer();

          const size_t global = RowA;
          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                      &global, NULL, 0, NULL, &gemmKernelEvent) );

          CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );

        }else if( !a_traspose && !b_traspose ){ // Transpose A: no, Transpose B: no

          transKernelIter = ColB >> 4;
          gemmKernelIter = ColA >> 4;

          // Transpose B
          SET_TRANS_KERNEL_ARG(RowB, ColB, clBufferB, clBufferB_T, transKernelIter );

          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                      &ColA, NULL, 0, NULL, &transKernelEvent[0]) );

          SET_GEMM_TN_KERNEL_ARG(RowA, ColA, ColB, clBufferA, clBufferB_T,
            clBufferC, ColA, float, gemmKernelIter);

          startTimer();

          const size_t global = RowA;
          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                      &global, NULL, 1, transKernelEvent, &gemmKernelEvent) );

          CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );
        }

        getPerformance();
        return CL_SUCCESS;
      }

    protected:

      // OpenCL memeory object
      cl_mem clBufferA;
      cl_mem clBufferA_T;
      cl_mem clBufferB;
      cl_mem clBufferB_T;
      cl_mem clBufferC;

      // Host memory data
      cl_float * clHostPtrA;
      cl_float * clHostPtrB;
      cl_float * clHostPtrC;

      // OpenCL events
      cl_event gemmKernelEvent;
      cl_event transKernelEvent[2];
      cl_event mapBufferEvents[2];
      cl_event unMapBufferEvents[2];

      // OpenCL program object
      cl_program clProgram;

      // OpenCL kernel object
      cl_kernel clGemmKernel;
      cl_kernel clTransKernel;

  };  // class clQualcommFP32Engine

  // clQualcommFP16Engine concrete class using Qualcomm GEMM example
  class clQualcommFP16Engine : public clQualcommFP32Engine{
    public:

      cl_int memLoad(typename functor::MatMulTypes<float>::out_type out){

        // Use the map function to return clBufferA pointer to the host <= blocking
        clHostPtrC = ( cl_float * ) clEnqueueMapBuffer(clQueue, clBufferC, CL_TRUE,
                                      CL_MAP_READ, 0, c_size, 0, NULL, NULL, NULL);

        // Read computed result back to host
        for( auto idx = 0 ; idx < RowC*ColC ; idx++){
          out.data()[idx] = clHostPtrC[idx];
        }

        // Release OpenCL resources
        CL_CHECK( clEnd() );

        // Return if the results are loaded to memory & OpenCL resources are released
        return CL_SUCCESS;
      }

      cl_int memInit(
        typename functor::MatMulTypes<float>::in_type in0,
        typename functor::MatMulTypes<float>::in_type in1)
      {

        // FP16 is half of the size of FP32
        a_size = a_size >> 1;
        b_size = b_size >> 1;

        // Use zero copy to avoid memeory copy
        // Matrix A
        clBufferA = clCreateBuffer(clCtx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      a_size, NULL, NULL);
        // Use the map function to return clBufferA pointer to the host <= non-blocking
        clHostFp16PtrA = ( cl_half * ) clEnqueueMapBuffer(clQueue, clBufferA, CL_FALSE,
                                        CL_MAP_WRITE, 0, a_size, 0, NULL,
                                        &mapBufferEvents[0], NULL);
        // Matrix B
        clBufferB = clCreateBuffer(clCtx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      b_size, NULL, NULL);
        // Use the map function to return clBufferA pointer to the host <= non-blocking
        clHostFp16PtrB = ( cl_half * ) clEnqueueMapBuffer(clQueue, clBufferB, CL_FALSE,
                                        CL_MAP_WRITE, 0, b_size, 0, NULL,
                                        &mapBufferEvents[1], NULL);

        // Create GPU buffer for transposed matrices only if needed
        if( a_traspose ){
          clBufferA_T = clCreateBuffer(clCtx, CL_MEM_HOST_NO_ACCESS, a_size, NULL, NULL);
        }
        if( !b_traspose ){
          clBufferB_T = clCreateBuffer(clCtx, CL_MEM_HOST_NO_ACCESS, b_size, NULL, NULL);
        }

        // Wait for completion
        CL_CHECK( clWaitForEvents(2, mapBufferEvents) );

        // Host update the buffer using pointer clHostFp16PtrA in host address space
        for( auto idx = 0 ; idx < RowA*ColA ; idx ++){
          clHostFp16PtrA[ idx ] = float_to_cl_half( in0.data()[idx] );
        }
        // Host update the buffer using pointer clHostFp16PtrB in host address space
        for( auto idx = 0 ; idx < RowB*ColB ; idx ++){
          clHostFp16PtrB[ idx ] = float_to_cl_half( in1.data()[idx] );
        }

        // Unmap the object -> Used in the OpenCL kernel
        CL_CHECK( clEnqueueUnmapMemObject( clQueue, clBufferA, (void*) clHostFp16PtrA,
                    0, NULL, &unMapBufferEvents[0] ) );

        // Unmap the object -> Used in the OpenCL kernel
        CL_CHECK( clEnqueueUnmapMemObject( clQueue, clBufferB, (void*) clHostFp16PtrB,
                    0, NULL, &unMapBufferEvents[1] ) );

        // Matrix C
        clBufferC = clCreateBuffer(clCtx, CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                      c_size, NULL, NULL);

        // Wait for completion
        CL_CHECK( clWaitForEvents(2, unMapBufferEvents) );
        return CL_SUCCESS;
      }

      cl_int loadFromBinaryCompute()
      {

        unsigned char* clKernelBinaryFile = NULL;
        size_t clKernelBinSize = 0;
        // Read compiled OpenCL kernel binary file from disk
        read_file(&clKernelBinaryFile, &clKernelBinSize, "matmul.bin" );

        // Create an OpenCL program object from binary
        clProgram = CL_CHECK_ERR( clCreateProgramWithBinary(clCtx, 1, &clDevice,
                                    &clKernelBinSize,
                                    (const unsigned char **)&clKernelBinaryFile,
                                    NULL, &_err) );

        // OpenCL build program
        CL_CHECK( clBuildProgram(clProgram, 1, &clDevice, NULL , NULL, NULL) );

        cl_ushort gemmKernelIter;
        cl_ushort transKernelIter;

        // Create OpenCL GEMM kernel object
        // clGemmKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatMul_TN_1D_Fp16_Half4" , &_err) );
        clGemmKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatMul_TN_1D_Fp16_Half8" , &_err) );
        // clGemmKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatMul_TN_1D_Fp16_Half16" , &_err) );

        // Create OpenCL Transpose kernel object
        // clTransKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatTrans_1D_Fp16_Half4" , &_err) );
        clTransKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatTrans_1D_Fp16_Half8" , &_err) );
        // clTransKernel = CL_CHECK_ERR( clCreateKernel(clProgram, "MatTrans_1D_Fp16_Half16" , &_err) );

        // Handle Matrices Transpose
        if( a_traspose && b_traspose ){ // Transpose A: yes, Transpose B: yes

          transKernelIter = ColA >> 3;
          gemmKernelIter = RowA >> 3;

          // Transpose A
          SET_TRANS_KERNEL_ARG(RowA, ColA, clBufferA, clBufferA_T, transKernelIter);

          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                      &RowA, NULL, 0, NULL, &transKernelEvent[0]) );

          SET_GEMM_TN_KERNEL_ARG(ColA, RowA, RowB, clBufferA_T, clBufferB,
            clBufferC, ColA, cl_half, gemmKernelIter);

          const size_t global = ColA;
          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                      &global, NULL, 1, transKernelEvent, &gemmKernelEvent) );

          CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );

        }else if( a_traspose && !b_traspose ){ // Transpose A: yes, Transpose B: no

          transKernelIter = ColA >> 3;
          gemmKernelIter = RowA >> 3;

          // Transpose A
          SET_TRANS_KERNEL_ARG(RowA, ColA, clBufferA, clBufferA_T, transKernelIter);

          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                      &RowA, NULL, 0, NULL, &transKernelEvent[0]) );

          transKernelIter = ColB >> 3;

          // Transpose B
          SET_TRANS_KERNEL_ARG(RowB, ColB, clBufferB, clBufferB_T, transKernelIter);

          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                      &RowB, NULL, 0, NULL, &transKernelEvent[1]) );

          SET_GEMM_TN_KERNEL_ARG(ColA, RowA, ColB, clBufferA_T, clBufferB_T,
            clBufferC, RowA, cl_half, gemmKernelIter);

          const size_t global = ColA;
          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                      &global, NULL, 2, transKernelEvent, &gemmKernelEvent) );

          CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );

        }else if( !a_traspose && b_traspose ){ // Transpose A: no, Transpose B: yes

          gemmKernelIter = ColA >> 3;

          // Transpose A
          SET_GEMM_TN_KERNEL_ARG(RowA, ColA, RowB, clBufferA, clBufferB,
            clBufferC, ColA, cl_half, gemmKernelIter);

          const size_t global = RowA;
          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                      &global, NULL, 0, NULL, &gemmKernelEvent) );

          CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );

        }else if( !a_traspose && !b_traspose ){ // Transpose A: no, Transpose B: no

          transKernelIter = ColB >> 3;
          gemmKernelIter = ColA >> 3;

          // Transpose B
          SET_TRANS_KERNEL_ARG(ColA, ColB, clBufferB, clBufferB_T, transKernelIter);

          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                      &ColA, NULL, 0, NULL, &transKernelEvent[0]) );

          SET_GEMM_TN_KERNEL_ARG(RowA, ColA, ColB, clBufferA, clBufferB_T,
            clBufferC, ColA, cl_half, gemmKernelIter);

          const size_t global = RowA;
          CL_CHECK( clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                      &global, NULL, 1, transKernelEvent, &gemmKernelEvent) );

          CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );
        }
        return CL_SUCCESS;
      }

    private:
      // Copied memory data
      cl_half * clHostFp16PtrA;
      cl_half * clHostFp16PtrB;

  };  // class clQualcommFP16Engine

  // clBLASTEngine concrete class using CLBLAST API
  class clBLASTEngine : public clMatMulEngine<float>{
    public:

      cl_int clEnd(){

        // Free OpenCL memory objects
        CL_CHECK( clReleaseMemObject(clBufferA) );
        CL_CHECK( clReleaseMemObject(clBufferB) );
        CL_CHECK( clReleaseMemObject(clBufferC) );

        // Free OpenCL command queue
        CL_CHECK( clReleaseCommandQueue(clQueue) );

        // Free OpenCL context
        CL_CHECK( clReleaseContext(clCtx) );

        // Free OpenCL events
        CL_CHECK( clReleaseEvent(gemmKernelEvent) );
        CL_CHECK( clReleaseEvent(writeBufferEvents[0]) );
        CL_CHECK( clReleaseEvent(writeBufferEvents[1]) );

        // Return CL_SUCCESS if all resources are released successfully
        return CL_SUCCESS;
      }

      cl_int memLoad(typename functor::MatMulTypes<float>::out_type out){

        // Read results
        CL_CHECK( clEnqueueReadBuffer(clQueue, clBufferC, CL_TRUE, 0, c_size,
                    out.data(), 0, NULL, NULL) );

        // Release OpenCL resources
        CL_CHECK( clEnd() );

        // Return if the results are loaded to memory & OpenCL resources are released
        return CL_SUCCESS;
      }

      cl_int memInit(
        typename functor::MatMulTypes<float>::in_type in0,
        typename functor::MatMulTypes<float>::in_type in1)
      {

        // Allocate memory buffers
        clBufferA = CL_CHECK_ERR( clCreateBuffer(clCtx, CL_MEM_READ_ONLY, a_size,
                                    NULL, &_err) );
        clBufferB = CL_CHECK_ERR( clCreateBuffer(clCtx, CL_MEM_READ_ONLY, b_size,
                                    NULL, &_err) );
        clBufferC = CL_CHECK_ERR( clCreateBuffer(clCtx, CL_MEM_READ_WRITE, c_size,
                                    NULL, &_err) );

        // Enqueue write buffer commands (acynchronous write)
        CL_CHECK( clEnqueueWriteBuffer(clQueue, clBufferA, CL_FALSE, 0, a_size,
                    in0.data(), 0, NULL, &writeBufferEvents[0]) );

        CL_CHECK( clEnqueueWriteBuffer(clQueue, clBufferB, CL_FALSE, 0, b_size,
                    in1.data(), 0, NULL, &writeBufferEvents[1]) );

        // Wait for completion
        CL_CHECK( clWaitForEvents(2, writeBufferEvents) );
        return CL_SUCCESS;
      }

      cl_int clBlastCompute()
      {
        // Whether Matrix A, B should be transposed
        auto MatATranspose = ( a_traspose == true ) ?
                              CLBlastTransposeYes : CLBlastTransposeNo;
        auto MatBTranspose = ( b_traspose == true ) ?
                              CLBlastTransposeYes : CLBlastTransposeNo;

        // Leading dimension of the input A matrix. This value must be greater than 0.
        size_t a_ld;

        // Leading dimension of the input B matrix. This value must be greater than 0.
        size_t b_ld;

        // When transpose_a == Transpose::kNo, then a_ld must be at least m,
        // otherwise a_ld must be at least k.
        if( MatATranspose == CLBlastTransposeYes ){
          a_ld = RowA;
        }else{
          a_ld = ColA;
        }

        // When transpose_b == Transpose::kNo, then b_ld must be at least k,
        // otherwise b_ld must be at least n.
        if( MatBTranspose == CLBlastTransposeYes ){
          b_ld = ColA;
        }else{
          b_ld = ColB;
        }

        // The value of c_ld must be at least m.
        const size_t c_ld = ColB;

        // Performs the matrix product C = alpha * A * B + beta * C
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Call the SGEMM routine.
        CLBlastStatusCode status = CLBlastSgemm(CLBlastLayoutRowMajor,
                                                MatATranspose, MatBTranspose,
                                                RowA, ColB, ColA,
                                                alpha,
                                                clBufferA, 0, a_ld,
                                                clBufferB, 0, b_ld,
                                                beta,
                                                clBufferC, 0, c_ld,
                                                &clQueue, &gemmKernelEvent);

        // Wait for completion
        if (status != CLBlastSuccess){
          LOG(ERROR) << "[CLBlast] Fail with code " << status;
          return CL_FALSE;
        }

        CL_CHECK( clWaitForEvents(1, &gemmKernelEvent) );
        return CL_SUCCESS;
      }

    protected:

      // OpenCL memeory object
      cl_mem clBufferA;
      cl_mem clBufferB;
      cl_mem clBufferC;

      // OpenCL events
      cl_event gemmKernelEvent;
      cl_event writeBufferEvents[2];

  };  // class clBLASTEngine

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

      clQualcommFP32Engine c = clQualcommFP32Engine();
      // clQualcommFP16Engine c = clQualcommFP16Engine();
      // clBLASTEngine c = clBLASTEngine();

      // OpenCL host & device side initializaiotn
      CL_CHECK( c.hostInit(in0, in1, out, dim_pair) );

      // debug info
      // c.debug(true);

      // OpenCL memeory object init & memory copy
      CL_CHECK( c.memInit(in0, in1) );

      // GEMM computation
      CL_CHECK( c.loadFromBinaryCompute() );
      // CL_CHECK( c.clBlastCompute() );

      // OpenCL memory load
      CL_CHECK( c.memLoad(out) );

      // Results
      // c.printMatrix(in0, in1, out);

    }
  };

}  // end namespace functor
}  // end namespace tensorflow

#endif  // MATMUL_CL_FUNCTOR_H_
