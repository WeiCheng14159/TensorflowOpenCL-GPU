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

#include "tensorflow/core/framework/common_shape_fns.h"
// #include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// Ops registration
namespace tensorflow {

  REGISTER_OP("Addcl")
      .Input("x: T")
      .Input("y: T")
      .Output("z: T")
      .Attr(
          "T: {half, bfloat16, float, double, uint8, int8, int16, int32, int64}")
      .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
      .Doc(R"doc(
  Add 2 Tensors z = x + y.
  x, y, z should all be the same size.

  )doc");
}  // namespace tensorflow

#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/platform/logging.h"
#include <algorithm>
#include <iterator>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#ifdef __APPLE__
  #include <OpenCL/cl.hpp>
#else
  #include "cl2.hpp"
#endif

// Platform getPlatform() {
//   /* Returns the first platform found. */
//   std::vector<Platform> all_platforms;
//   Platform::get(&all_platforms);
//
//   if (all_platforms.size()==0) {
//     LOG(ERROR) << "No platforms found. Check OpenCL installation!\n";
//   }
//   return all_platforms[0];
// }
//
// Device getDevice(Platform platform, int i, bool display=true) {
//   /* Returns the deviced specified by the index i on platform.
//    * If display is true, then all of the platforms are listed.
//    */
//   std::vector<Device> all_devices;
//   platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
//   if(all_devices.size()==0){
//       LOG(ERROR) << "No devices found. Check OpenCL installation!\n";
//   }
//
//   if (display) {
//     char buff[100];
//     for (int j=0; j<all_devices.size(); j++)
//       snprintf( buff, sizeof(buff), "Device %d: %s\n", j,
//         all_devices[j].getInfo<CL_DEVICE_NAME>().c_str());
//       LOG(INFO) << buff;
//   }
//   return all_devices[i];
// }

// Kernel implementation
namespace tensorflow {

REGISTER5(BinaryOp, CPU, "Addcl", functor::add, float, Eigen::half, double,
          int32, int64);

class AddclOp : public OpKernel {
  public:
    explicit AddclOp(OpKernelConstruction* context) : OpKernel(context) {
      // Check OpenCL context

      // throw an excetion if the CL object check fail

    }

    void Compute(OpKernelContext* context) override {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);
      auto input = input_tensor.flat<int32>();

      // Create an output tensor
      Tensor* output_tensor = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                       &output_tensor));
      auto output = output_tensor->template flat<int32>();

      // Set all but the first element of the output tensor to 0.
      const int N = input.size();
      for (int i = 1; i < N; i++) {
        output(i) = output(i) + 1;
      }
    }
}; // class AddclOp

REGISTER_KERNEL_BUILDER(Name("Addcl").Device(DEVICE_CPU), AddclOp);

}  // namespace tensorflow
