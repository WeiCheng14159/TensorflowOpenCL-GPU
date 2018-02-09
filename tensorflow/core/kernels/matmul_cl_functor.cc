#ifdef TEST_CL

#include "matmul_cl_functor.h"

#include "tensorflow/core/platform/logging.h"

using namespace cl;

MatMulClFunctor::MatMulClFunctor(){
    init();
}

void MatMulClFunctor::init(){
  Platform plat = getPlatform();
  Device device = getDevice(plat, 0);
  Context context({device});
  Program::Sources sources;
  sources.push_back({matmul_kernel.c_str(), matmul_kernel.length()});

  program = Program(context, sources);
  if (program.build({device}) != CL_SUCCESS) {
      LOG(ERROR) << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
  }
  // buffer_A = Buffer(context, CL_MEM_READ_WRITE, sizeof(int) * n);
  queue = CommandQueue(context, device);
}

void MatMulClFunctor::build_kernel(){

}

void MatMulClFunctor::mem_setup(){

}

void MatMulClFunctor::submit(){

}

Device MatMulClFunctor::getDevice(Platform platform, int i, bool display) {
  /* Returns the deviced specified by the index i on platform.
   * If display is true, then all of the platforms are listed.
   */
  std::vector<Device> all_devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size()==0){
      LOG(ERROR) << "No devices found. Check OpenCL installation!\n";
  }

  if (display) {
    for (int j=0; j<all_devices.size(); j++)
      LOG(INFO) << all_devices[j].getInfo<CL_DEVICE_NAME>().c_str()  << "\n";
  }
  return all_devices[i];
}

Platform MatMulClFunctor::getPlatform() {
  /* Returns the first platform found. */
  std::vector<Platform> all_platforms;
  Platform::get(&all_platforms);

  if (all_platforms.size()==0)
    LOG(ERROR) << "No platforms found. Check OpenCL installation!\n";

  return all_platforms[0];
}

#endif // TEST_CL
