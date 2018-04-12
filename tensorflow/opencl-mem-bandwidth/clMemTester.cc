#include "CL/cl.h"
#include "Timer.h"
#include "clMemTester.h"
#include <iostream>

// clMemTester constructor
clMemTester::clMemTester(int num){
  numTests = num;
}

// Init OpenCL objects
cl_int clMemTester::init()
{
  // OpenCL error code init
  err = CL_SUCCESS;

  // Query platforms
  err = clGetPlatformIDs(1, &platform, NULL);
  if( err != CL_SUCCESS )
    return err;

  // Query devices
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &clDevice, NULL);
  if( err != CL_SUCCESS )
    return err;

  // Create context
  clCtx = clCreateContext(NULL, 1, &clDevice, NULL, NULL, NULL);

  // Create command clQueue
  clQueue = clCreateCommandQueue(clCtx, clDevice, 0, NULL);

  // Timer init
  Timer timer = Timer();

  return CL_SUCCESS;
}

// Release all OpenCL related resourcse
cl_int clMemTester::clEnd(){
  clReleaseCommandQueue(clQueue);
  clReleaseContext(clCtx);
  return CL_SUCCESS;
}

// Host to device memory bandwidth test
cl_int clMemTester::HostToDevice( unsigned long int numBytes )
{
  // Create host buffer
  char * hostBufPtr = new char [ numBytes ];
  for ( auto i = 0; i < numBytes; i++ )
  {
      hostBufPtr[i] = (i & 0xff);
  }

  // err code init
  err = CL_SUCCESS;

  // Create device buffer
  cl_mem deviceBuffer = clCreateBuffer( clCtx, CL_MEM_READ_WRITE, numBytes, NULL, &err );
  if ( err != CL_SUCCESS )
  {
      std::cerr << "clCreateBuffer fail with code " << err;
      delete [] hostBufPtr;
      return err;
  }

  clFinish( clQueue );

  timer.start();

  // Write host -> device
  for ( size_t i = 0; i < numTests; i++ )
  {
      // Asynchronous write
      err = clEnqueueWriteBuffer( clQueue, deviceBuffer, CL_FALSE, 0, numBytes,
        hostBufPtr, 0, NULL, NULL );
      if (err != CL_SUCCESS )
      {
          std::cerr << "Error writing device buffer";
          clReleaseMemObject( deviceBuffer );
          delete [] hostBufPtr;
          return err;
      }
  }

  // Finish any outstanding writes
  clFinish( clQueue );

  computeBandwidth( numBytes, timer.read_us() );
  delete [] hostBufPtr;
  clReleaseMemObject( deviceBuffer );
  return CL_SUCCESS;
}

// Device to host memory bandwidth test
cl_int clMemTester::DeviceToHost( unsigned long int numBytes )
{
  // Create host buffer
  char * hostBufPtr = new char [ numBytes ];
  for ( auto i = 0; i < numBytes; i++ )
  {
      hostBufPtr[i] = (i & 0xff);
  }

  // err code init
  err = CL_SUCCESS;

  // Copy the contents of the host buffer into a device buffer
  cl_mem deviceBuffer = clCreateBuffer( clCtx,
    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numBytes, hostBufPtr, &err );
  if ( err != CL_SUCCESS )
  {
      std::cerr << "clCreateBuffer fail with code " << err;
      delete [] hostBufPtr;
      return err;
  }

  clFinish( clQueue );

  timer.start();

  // Read from device -> host
  for ( size_t i = 0; i < numTests; i++ )
  {
      // Asynchronous read
      err = clEnqueueReadBuffer( clQueue, deviceBuffer, CL_FALSE, 0, numBytes,
        hostBufPtr, 0, NULL, NULL );
      if (err != CL_SUCCESS )
      {
          std::cerr << "Error writing device buffer";
          clReleaseMemObject( deviceBuffer );
          delete [] hostBufPtr;
          return err;
      }
  }

  // Finish any outstanding writes
  clFinish( clQueue );

  computeBandwidth( numBytes, timer.read_us() );
  delete [] hostBufPtr;
  clReleaseMemObject( deviceBuffer );
  return CL_SUCCESS;
}

// Device to device memory bandwidth test
cl_int clMemTester::DeviceToDevice( unsigned long int numBytes )
{
  // Create host buffer
  char * hostBufPtr = new char [ numBytes ];
  for ( auto i = 0; i < numBytes; i++ )
  {
      hostBufPtr[i] = (i & 0xff);
  }

  // err code init
  err = CL_SUCCESS;

  // Copy the contents of the host buffer into a device buffer
  cl_mem deviceBufferSrc = clCreateBuffer( clCtx,
    CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numBytes, hostBufPtr, &err );
  if ( err != CL_SUCCESS )
  {
      std::cerr << "clCreateBuffer fail with code " << err;
      clReleaseMemObject( deviceBufferSrc );
      delete [] hostBufPtr;
      return err;
  }

  // Create another device buffer to copy into
  cl_mem deviceBufferDst = clCreateBuffer( clCtx, CL_MEM_READ_WRITE, numBytes,
    NULL, &err );
  if ( err != CL_SUCCESS )
  {
      std::cerr << "clCreateBuffer fail with code " << err;
      clReleaseMemObject( deviceBufferDst );
      delete [] hostBufPtr;
      return err;
  }

  clFinish( clQueue );

  timer.start();

  // Copy from device -> device
  for ( size_t i = 0; i < numTests; i++ )
  {
      // Asynchronous write
      err = clEnqueueCopyBuffer( clQueue, deviceBufferSrc, deviceBufferDst,
        0, 0, numBytes, 0, NULL, NULL );
      if (err != CL_SUCCESS )
      {
          std::cerr << "Error copying device buffer";
          clReleaseMemObject( deviceBufferSrc );
          clReleaseMemObject( deviceBufferDst );
          delete [] hostBufPtr;
          return err;
      }
  }

  // Finish any outstanding writes
  clFinish( clQueue );

  computeBandwidth( numBytes, timer.read_us() );
  delete [] hostBufPtr;
  clReleaseMemObject( deviceBufferSrc );
  clReleaseMemObject( deviceBufferDst );
  return CL_SUCCESS;
}

// Memory bandwidth calculator
void clMemTester::computeBandwidth(size_t numOfBytes, const double& time_us){

  double MB = numOfBytes / (1024*1024);
  printf("Writing %.2f MB, Bandwidth = %.2f GB/s\n", MB, MB * numTests * 1e6 / time_us / 1024);
}
