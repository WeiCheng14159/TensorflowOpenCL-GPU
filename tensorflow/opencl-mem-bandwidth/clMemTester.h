#ifndef CL_MEM_TESTER_H
#define CL_MEM_TESTER_H

#include "CL/cl.h"
#include "Timer.h"

class clMemTester{
public:

  // clMemTester constructor
  clMemTester(int num);

  // Init OpenCL objects
  cl_int init();

  // Release all OpenCL related resourcse
  cl_int clEnd();

  // Host to device memory bandwidth test
  cl_int HostToDevice( unsigned long int numBytes );

  // Device to host memory bandwidth test
  cl_int DeviceToHost( unsigned long int numBytes );

  // Device to device memory bandwidth test
  cl_int DeviceToDevice( unsigned long int numBytes );

  // Memory bandwidth calculator 
  void computeBandwidth(size_t numOfBytes, const double& time_us);

private:
  cl_platform_id platform;
  cl_device_id clDevice;
  cl_context clCtx;
  cl_command_queue clQueue;
  cl_int err = CL_SUCCESS;

  Timer timer;
  int numTests;

};

#endif
