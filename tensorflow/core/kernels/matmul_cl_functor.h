#ifdef TEST_CL
#ifndef MATMUL_CL_FUNCTOR_H_
#define MATMUL_CL_FUNCTOR_H_

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 100
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "cl2.hpp"

class MatMulClFunctor {
public:

  MatMulClFunctor();

  void init();

  void build_kernel();

  void mem_setup();

  void submit();

private:
  cl::Platform plat;
  cl::Device device;
  cl::Context context;
  cl::Program program;
  cl::CommandQueue queue;
  cl::Buffer buffer_A;
  std::string matmul_kernel =
    {R"CLC(
        void kernel multiply_by(global float* A, global float* B, global float* C) {
          C[get_global_id(0)] = A[get_global_id(0)] + B[get_global_id(0)];
        }
    )CLC"};

  cl::Device getDevice(cl::Platform platform, int i, bool display=false);

  cl::Platform getPlatform();

};  // class MatMulClFunctor

#endif  // MATMUL_CL_FUNCTOR_H_
#endif  // TEST_CL
