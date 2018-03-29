// clMatMulEngine<float>
//     |
//     v
// clQualcommFP32Engine <---- binaryLoaderInterface
//
// clMatMulEngine<float>
//     |
//     v
// clBLASTEngine

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

        int matrixSizeLimit = 0xffff;

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

        // OpenCL error code init
        err = CL_SUCCESS;

        // Query platforms
        err = clGetPlatformIDs(1, &platform, NULL);
        if( err != CL_SUCCESS ){
          LOG(INFO) << "clGetPlatformIDs fail with code " << err;
          return err;
        }

        // Query devices
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &clDevice, NULL);
        if( err != CL_SUCCESS ){
          LOG(INFO) << "clGetDeviceIDs fail with code " << err;
          return err;
        }

        // Create context
        clCtx = clCreateContext(NULL, 1, &clDevice, NULL, NULL, &err);
        if( err != CL_SUCCESS ){
          LOG(INFO) << "clCreateContext fail with code " << err;
          return err;
        }

        // Create command clQueue
        clQueue = clCreateCommandQueue(clCtx, clDevice, 0, &err);
        if( err != CL_SUCCESS ){
          LOG(INFO) << "clCreateCommandQueue fail with code " << err;
          return err;
        }

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
      virtual cl_int memInit(
        typename functor::MatMulTypes<T>::in_type in0,
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
      cl_int err = CL_SUCCESS;

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

      void debugOpenclKernel(cl_kernel cl_kernel, cl_device_id cl_device){

        // Kernel info
        size_t wgSize = 0;
        size_t compiledWgSize[3];
        cl_ulong localMemSize = 0;
        size_t perfHint;
        cl_ulong privateMemSize = 0;

        if(
          clGetKernelWorkGroupInfo(cl_kernel, cl_device, CL_KERNEL_WORK_GROUP_SIZE,
            sizeof(size_t), &wgSize, NULL) != CL_SUCCESS                      ||
          clGetKernelWorkGroupInfo(cl_kernel, cl_device, CL_KERNEL_COMPILE_WORK_GROUP_SIZE,
            3 * sizeof(size_t), &compiledWgSize, NULL) != CL_SUCCESS          ||
          clGetKernelWorkGroupInfo(cl_kernel, cl_device, CL_KERNEL_LOCAL_MEM_SIZE,
            sizeof(cl_ulong), &localMemSize, NULL) != CL_SUCCESS              ||
          clGetKernelWorkGroupInfo(cl_kernel, cl_device, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
            sizeof(size_t), &perfHint, NULL) != CL_SUCCESS                    ||
          clGetKernelWorkGroupInfo(cl_kernel, cl_device, CL_KERNEL_PRIVATE_MEM_SIZE,
            sizeof(cl_ulong), &privateMemSize, NULL) != CL_SUCCESS
          ){
            printf("debugOpenclKernel fail\n");
          }else{
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
      }
  };  // class binaryLoaderInterface

  // clQualcommFP32Engine concrete class using Qualcomm GEMM example
  class clQualcommFP32Engine : public binaryLoaderInterface, public clMatMulEngine<float>{
    public:

      cl_int clEnd(){

        // Free OpenCL memory objects
        clReleaseMemObject(clBufferA);
        clReleaseMemObject(clBufferA_T);
        clReleaseMemObject(clBufferB);
        clReleaseMemObject(clBufferB_T);
        clReleaseMemObject(clBufferC);

        // Free OpenCL kernel
        clReleaseKernel(clGemmKernel);
        clReleaseKernel(clTransKernel);

        // Free OpenCL program
        clReleaseProgram(clProgram);

        // Free OpenCL command queue
        clReleaseCommandQueue(clQueue);

        // Free OpenCL context
        clReleaseContext(clCtx);

        // Free OpenCL events
        clReleaseEvent(transKernelEvent[0]);
        clReleaseEvent(transKernelEvent[1]);
        clReleaseEvent(gemmKernelEvent);
        clReleaseEvent(mapBufferEvents[0]);
        clReleaseEvent(mapBufferEvents[1]);
        clReleaseEvent(unMapBufferEvents[0]);
        clReleaseEvent(unMapBufferEvents[1]);

        // Return CL_SUCCESS if all resources are released successfully
        return CL_SUCCESS;
      }

      cl_int memLoad(typename functor::MatMulTypes<float>::out_type out){

        // Init cl err code
        err = CL_SUCCESS;

        // Use the map function to return clBufferA pointer to the host <= blocking
        clHostPtrC = ( cl_float * ) clEnqueueMapBuffer(clQueue, clBufferC, CL_TRUE,
                          CL_MAP_READ, 0, c_size, 0, NULL, NULL, &err);

        // Read results
        if( err == CL_SUCCESS ){
          // Read computed result back to host
          for( auto idx = 0 ; idx < RowC*ColC ; idx++){
            out.data()[idx] = clHostPtrC[idx];
          }
        }else{
          LOG(ERROR) << "Host-side pointer for matrix C is invalid";
          return CL_FALSE;
        }

        // Release OpenCL resources
        err = clEnd();
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clEnd() fail with code " << err;
          return err;
        }

        // Return if the results are loaded to memory & OpenCL resources are released
        return CL_SUCCESS;
      }

      cl_int memInit(
        typename functor::MatMulTypes<float>::in_type in0,
        typename functor::MatMulTypes<float>::in_type in1)
      {
        // Init cl err code
        err = CL_SUCCESS;

        // Use zero copy to avoid memeory copy
        // Matrix A
        clBufferA = clCreateBuffer(clCtx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                          a_size, NULL, NULL);
        // Use the map function to return clBufferA pointer to the host <= non-blocking
        clHostPtrA = ( cl_float * ) clEnqueueMapBuffer(clQueue, clBufferA, CL_FALSE,
                          CL_MAP_WRITE, 0, a_size, 0, NULL, &mapBufferEvents[0], NULL);
        // Matrix B
        clBufferB = clCreateBuffer(clCtx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                          b_size, NULL, NULL);
        // Use the map function to return clBufferA pointer to the host <= non-blocking
        clHostPtrB = ( cl_float * ) clEnqueueMapBuffer(clQueue, clBufferB, CL_FALSE,
                          CL_MAP_WRITE, 0, b_size, 0, NULL, &mapBufferEvents[1], NULL);

        // Create GPU buffer for transposed matrices only if needed
        if( a_traspose ){
          clBufferA_T = clCreateBuffer(clCtx, CL_MEM_HOST_NO_ACCESS, a_size, NULL, NULL);
        }
        if( !b_traspose ){
          clBufferB_T = clCreateBuffer(clCtx, CL_MEM_HOST_NO_ACCESS, b_size, NULL, NULL);
        }

        // Wait for completion
        clWaitForEvents(2, mapBufferEvents);

        // Host update the buffer using pointer clHostPtrA in host address space
        for( auto idx = 0 ; idx < RowA*ColA ; idx ++){
          clHostPtrA[ idx ] = in0.data()[idx];
        }
        // Host update the buffer using pointer clHostPtrB in host address space
        for( auto idx = 0 ; idx < RowB*ColB ; idx ++){
          clHostPtrB[ idx ] = in1.data()[idx];
        }

        // Unmap the object -> Used in the OpenCL kernel
        err = clEnqueueUnmapMemObject( clQueue, clBufferA, (void*) clHostPtrA, 0, NULL,
                                      &unMapBufferEvents[0] );
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clEnqueueUnmapMemObject fail with code " << err;
          return err;
        }

        // Unmap the object -> Used in the OpenCL kernel
        err = clEnqueueUnmapMemObject( clQueue, clBufferB, (void*) clHostPtrB, 0, NULL,
                                      &unMapBufferEvents[1] );
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clEnqueueUnmapMemObject fail with code " << err;
          return err;
        }

        // Matrix C
        clBufferC = clCreateBuffer(clCtx, CL_MEM_HOST_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                          c_size, NULL, NULL);

        // Wait for completion
        clWaitForEvents(2, unMapBufferEvents);
        return CL_SUCCESS;
      }

      cl_int loadFromBinaryCompute()
      {
        // Init cl err code
        err = CL_SUCCESS;

        unsigned char* clKernelBinaryFile = NULL;
        size_t clKernelBinSize = 0;
        // Read compiled OpenCL kernel binary file from disk
        read_file(&clKernelBinaryFile, &clKernelBinSize, "matmul.bin" );

        // Create an OpenCL program object from binary
        clProgram =
          clCreateProgramWithBinary(clCtx, 1, &clDevice, &clKernelBinSize,
                                  (const unsigned char **)&clKernelBinaryFile,
                                  NULL, &err);
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clCreateProgramWithBinary fail with code " << err;
          return err;
        }

        // OpenCL build program
        err = clBuildProgram(clProgram, 1, &clDevice, NULL , NULL, NULL);
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clBuildProgram fail with code " << err;
          return err;
        }

        // Create OpenCL GEMM kernel obj
        // clGemmKernel = clCreateKernel(clProgram, "MatMul_TN_1D_Fp32_Float4" , &err);
        // clGemmKernel = clCreateKernel(clProgram, "MatMul_TN_1D_Fp32_Float8" , &err);
        clGemmKernel = clCreateKernel(clProgram, "MatMul_TN_1D_Fp32_Float16" , &err);
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clCreateKernel fail with code " << err;
          return err;
        }

        // Create OpenCL Transpose kernel obj
        // clTransKernel = clCreateKernel(clProgram, "MatTrans_1D_Fp32_Float4" , &err);
        // clTransKernel = clCreateKernel(clProgram, "MatTrans_1D_Fp32_Float8" , &err);
        clTransKernel = clCreateKernel(clProgram, "MatTrans_1D_Fp32_Float16" , &err);
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clCreateKernel fail with code " << err;
          return err;
        }

        if( a_traspose && b_traspose ){ // Transpose A: yes, Transpose B: yes

          if (
            clSetKernelArg(clTransKernel, 0, sizeof(cl_ushort), &RowA) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 1, sizeof(cl_ushort), &ColA) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 2, sizeof(cl_mem), &clBufferA) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 3, sizeof(cl_mem), &clBufferA_T) != CL_SUCCESS
          ){
            LOG(ERROR) << "clSetKernelArg fail";
            return CL_FALSE;
          }

          err = clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                                       &RowA, NULL, 0, NULL, &transKernelEvent[0]);
          if( err != CL_SUCCESS ){
            LOG(ERROR) << "clEnqueueNDRangeKernel fail with code " << err;
            return err;
          }

          if (
            clSetKernelArg(clGemmKernel, 0, sizeof(cl_ushort), &ColA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 1, sizeof(cl_ushort), &RowA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 2, sizeof(cl_ushort), &RowB) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 3, sizeof(cl_mem), &clBufferA_T) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 4, sizeof(cl_mem), &clBufferB) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 5, sizeof(cl_mem), &clBufferC) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 6, ColA * sizeof(float), NULL) != CL_SUCCESS
          ){
            LOG(ERROR) << "clSetKernelArg fail";
            return CL_FALSE;
          }

          const size_t global = ColA;
          err = clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                                       &global, NULL, 1, transKernelEvent, &gemmKernelEvent);
          if( err != CL_SUCCESS ){
            LOG(ERROR) << "clEnqueueNDRangeKernel fail with code " << err;
            return err;
          }

          clWaitForEvents(1, &gemmKernelEvent);

        }else if( a_traspose && !b_traspose ){ // Transpose A: yes, Transpose B: no

          if (
            clSetKernelArg(clTransKernel, 0, sizeof(cl_ushort), &RowA) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 1, sizeof(cl_ushort), &ColA) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 2, sizeof(cl_mem), &clBufferA) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 3, sizeof(cl_mem), &clBufferA_T) != CL_SUCCESS
          ){
            LOG(ERROR) << "clSetKernelArg fail";
            return CL_FALSE;
          }

          err = clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                                       &RowA, NULL, 0, NULL, &transKernelEvent[0]);
          if( err != CL_SUCCESS ){
            LOG(ERROR) << "clEnqueueNDRangeKernel fail with code " << err;
            return err;
          }

          if (
            clSetKernelArg(clTransKernel, 0, sizeof(cl_ushort), &RowB) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 1, sizeof(cl_ushort), &ColB) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 2, sizeof(cl_mem), &clBufferB) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 3, sizeof(cl_mem), &clBufferB_T) != CL_SUCCESS
          ){
            LOG(ERROR) << "clSetKernelArg fail";
            return CL_FALSE;
          }

          err = clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                                       &RowB, NULL, 0, NULL, &transKernelEvent[1]);
          if( err != CL_SUCCESS ){
            LOG(ERROR) << "clEnqueueNDRangeKernel fail with code " << err;
            return err;
          }

          if (
            clSetKernelArg(clGemmKernel, 0, sizeof(cl_ushort), &ColA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 1, sizeof(cl_ushort), &RowA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 2, sizeof(cl_ushort), &ColB) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 3, sizeof(cl_mem), &clBufferA_T) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 4, sizeof(cl_mem), &clBufferB_T) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 5, sizeof(cl_mem), &clBufferC) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 6, RowA * sizeof(float), NULL) != CL_SUCCESS
          ){
            LOG(ERROR) << "clSetKernelArg fail";
            return CL_FALSE;
          }

          const size_t global = ColA;
          err = clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                                       &global, NULL, 2, transKernelEvent, &gemmKernelEvent);
          if( err != CL_SUCCESS ){
            LOG(ERROR) << "clEnqueueNDRangeKernel fail with code " << err;
            return err;
          }

          clWaitForEvents(1, &gemmKernelEvent);

        }else if( !a_traspose && b_traspose ){ // Transpose A: no, Transpose B: yes

          if (
            clSetKernelArg(clGemmKernel, 0, sizeof(cl_ushort), &RowA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 1, sizeof(cl_ushort), &ColA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 2, sizeof(cl_ushort), &RowB) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 3, sizeof(cl_mem), &clBufferA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 4, sizeof(cl_mem), &clBufferB) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 5, sizeof(cl_mem), &clBufferC) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 6, ColA * sizeof(float), NULL) != CL_SUCCESS
          ){
            LOG(ERROR) << "clSetKernelArg fail";
            return CL_FALSE;
          }

          const size_t global = RowA;
          err = clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                                       &global, NULL, 0, NULL, &gemmKernelEvent);
          if( err != CL_SUCCESS ){
            LOG(ERROR) << "clEnqueueNDRangeKernel fail with code " << err;
            return err;
          }

          clWaitForEvents(1, &gemmKernelEvent);

        }else if( !a_traspose && !b_traspose ){ // Transpose A: no, Transpose B: no

          if (
            clSetKernelArg(clTransKernel, 0, sizeof(cl_ushort), &ColA) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 1, sizeof(cl_ushort), &ColB) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 2, sizeof(cl_mem), &clBufferB) != CL_SUCCESS ||
            clSetKernelArg(clTransKernel, 3, sizeof(cl_mem), &clBufferB_T) != CL_SUCCESS
          ){
            LOG(ERROR) << "clSetKernelArg fail";
            return CL_FALSE;
          }

          err = clEnqueueNDRangeKernel(clQueue, clTransKernel, 1, NULL,
                                       &ColA, NULL, 0, NULL, &transKernelEvent[0]);
          if( err != CL_SUCCESS ){
            LOG(ERROR) << "clEnqueueNDRangeKernel fail with code " << err;
            return err;
          }

          if (
            clSetKernelArg(clGemmKernel, 0, sizeof(cl_ushort), &RowA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 1, sizeof(cl_ushort), &ColA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 2, sizeof(cl_ushort), &ColB) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 3, sizeof(cl_mem), &clBufferA) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 4, sizeof(cl_mem), &clBufferB_T) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 5, sizeof(cl_mem), &clBufferC) != CL_SUCCESS ||
            clSetKernelArg(clGemmKernel, 6, ColA * sizeof(float), NULL) != CL_SUCCESS
          ){
            LOG(ERROR) << "clSetKernelArg fail";
            return CL_FALSE;
          }

          const size_t global = RowA;
          err = clEnqueueNDRangeKernel(clQueue, clGemmKernel, 1, NULL,
                                       &global, NULL, 1, transKernelEvent, &gemmKernelEvent);
          if( err != CL_SUCCESS ){
            LOG(ERROR) << "clEnqueueNDRangeKernel fail with code " << err;
            return err;
          }

          clWaitForEvents(1, &gemmKernelEvent);
        }
        return CL_SUCCESS;
      }

    private:

      // OpenCL memeory object
      cl_mem clBufferA;
      cl_mem clBufferA_T;
      cl_mem clBufferB;
      cl_mem clBufferB_T;
      cl_mem clBufferC;

      // Copied memory data
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


  // clBLASTEngine concrete class using CLBLAST API
  class clBLASTEngine : public clMatMulEngine<float>{
    public:

      cl_int clEnd(){

        // Free OpenCL memory objects
        clReleaseMemObject(clBufferA);
        clReleaseMemObject(clBufferB);
        clReleaseMemObject(clBufferC);

        // Free OpenCL command queue
        clReleaseCommandQueue(clQueue);

        // Free OpenCL context
        clReleaseContext(clCtx);

        // Free OpenCL events
        clReleaseEvent(gemmKernelEvent);
        clReleaseEvent(writeBufferEvents[0]);
        clReleaseEvent(writeBufferEvents[1]);

        // Return CL_SUCCESS if all resources are released successfully
        return CL_SUCCESS;
      }

      cl_int memLoad(typename functor::MatMulTypes<float>::out_type out){

        // Init cl err code
        err = CL_SUCCESS;

        // Read results
        err = clEnqueueReadBuffer(clQueue, clBufferC, CL_TRUE, 0, c_size, out.data(), 0, NULL, NULL);
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clEnqueueReadBuffer fail with code " << err;
          return err;
        }

        // Release OpenCL resources
        err = clEnd();
        if( err != CL_SUCCESS ){
          LOG(ERROR) << "clEnd() fail with code " << err;
          return err;
        }

        // Return if the results are loaded to memory & OpenCL resources are released
        return CL_SUCCESS;
      }

      cl_int memInit(
        typename functor::MatMulTypes<float>::in_type in0,
        typename functor::MatMulTypes<float>::in_type in1)
      {

        // Init cl err code
        err = CL_SUCCESS;

        // Allocate memory buffers
        clBufferA = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, a_size, NULL, NULL);
        clBufferB = clCreateBuffer(clCtx, CL_MEM_READ_ONLY, b_size, NULL, NULL);
        clBufferC = clCreateBuffer(clCtx, CL_MEM_READ_WRITE, c_size, NULL, NULL);

        // Enqueue write buffer commands (acynchronous write)
        err = clEnqueueWriteBuffer(clQueue, clBufferA, CL_FALSE, 0, a_size, in0.data(),
                                   0, NULL, &writeBufferEvents[0]);
        if( err != CL_SUCCESS ){ return err; }

        err = clEnqueueWriteBuffer(clQueue, clBufferB, CL_FALSE, 0, b_size, in1.data(),
                                   0, NULL, &writeBufferEvents[1]);
        if( err != CL_SUCCESS ){ return err; }

        // Wait for completion
        clWaitForEvents(2, writeBufferEvents);
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

        clWaitForEvents(1, &gemmKernelEvent);
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

      // clBLASTEngine c = clBLASTEngine();
      clQualcommFP32Engine c = clQualcommFP32Engine();

      // Init cl status
      cl_int status = CL_SUCCESS;

      // OpenCL host & device side initializaiotn
      status = c.hostInit(in0, in1, out, dim_pair);
      if( status != CL_SUCCESS ){
        LOG(ERROR) << "CL init fail with code " << status;
      }

      // debug info
      // c.debug(true);

      // OpenCL memeory obj init & memory copy
      status = c.memInit(in0, in1);
      if( status != CL_SUCCESS ){
        LOG(ERROR) << "CL mem init fail with code " << status;
      }

      // GEMM computation
      status = c.loadFromBinaryCompute();
      // status = c.clBlastCompute();
      if( status != CL_SUCCESS ){
        LOG(ERROR) << "CL compute fail with code " << status;
      }

      // OpenCL memory load
      status = c.memLoad(out);
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
