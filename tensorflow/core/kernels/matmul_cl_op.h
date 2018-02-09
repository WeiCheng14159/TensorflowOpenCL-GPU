#ifdef TEST_CL

#ifndef TENSORFLOW_KERNELS_MATMUL_CL_OP_H_
#define TENSORFLOW_KERNELS_MATMUL_CL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"

#include "tensorflow/core/kernels/matmul_cl_functor.h"

namespace tensorflow {
namespace functor {
// MatMul OpenCL functor, the original functor is in matmul_op.h
template <typename Device, typename In0, typename In1, typename Out,
          typename DimPair>
void MatMulCL(const Device& d, Out out, In0 in0, In1 in1,
            const DimPair& dim_pair) {
  out.device(d) = in0.contract(in1, dim_pair);
  LOG(INFO) << "matmul running\n";
  MatMulClFunctor m = MatMulClFunctor();
  m.init();
}

}  // end namespace functor
}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MATMUL_CL_OP_H_
#endif  // TEST_CL
