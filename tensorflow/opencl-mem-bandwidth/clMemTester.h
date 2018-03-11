#ifndef CL_MEM_TESTER_H
#define CL_MEM_TESTER_H

#include "CL/cl.h"
#include "Timer.h"

class clMemTester{
public:
  clMemTester(int num);

  cl_int init();

  cl_int clEnd();

  cl_int HostToDevice( unsigned long int numBytes );

  cl_int DeviceToHost( unsigned long int numBytes );

  cl_int DeviceToDevice( unsigned long int numBytes );

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
