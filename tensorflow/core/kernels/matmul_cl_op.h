/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_MATMUL_CL_OP_H_
#define TENSORFLOW_KERNELS_MATMUL_CL_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace functor {
// MatMul OpenCL functor, the original functor is in matmul_op.h
template <typename Device, typename In0, typename In1, typename Out,
          typename DimPair>
void MatMulCL(const Device& d, Out out, In0 in0, In1 in1,
            const DimPair& dim_pair) {
  LOG(INFO) << "Debug kernel" << std::endl;
  out.device(d) = in0.contract(in1, dim_pair);
  LOG(INFO) << "dim_pair[0]" << dim_pair[0].first << std::endl;
  LOG(INFO) << "dim_pair[0]" << dim_pair[0].second << std::endl;
  // LOG(INFO) << dim_pair << std::endl;
}

}  // end namespace functor
}  // end namespace tensorflow

#define CL_HPP_TARGET_OPENCL_VERSION 200
#include "cl2.hpp"

namespace cl {
  class MatMulCL {
  public:

  private:
    
  }
}

#endif  // TENSORFLOW_KERNELS_MATMUL_CL_OP_H_
