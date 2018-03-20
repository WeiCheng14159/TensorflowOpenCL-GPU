# Test Tensorflow MatMul performance on Adreno 540 [Performance mode]

## 1. Test Tensorflow MatMul Ops performance on Qualcomm Snapdragon 835 platform.
The measured time below is the average of 10 TF MatMul runs. Notice the execution time measured here includes
the TF processing overhead, the results for pure GEMM operation can be found in `../opencl-clblast`
directory.   

## 2. Libraries tested:
  ### 1) Eigen:
    The default TF compute engine for Android devices. Currently, only CPU version is
    available. By default, the TF runtime assumes there're 8 symmetric cores on the devices.
  ### 2) Simple MatMul OpenCL kernel v1:
    The simplest MatMul OpenCL kernel. This provides a baseline for performance improvement.
  ### 3) Local memory MatMul OpenCL kernel v2:
    A slightly more complicated OpenCL kernel using local memory. Parameters for this
    configurations are work-group size, work-group dimension, local memory size.
  ### 4) Default CLBlast BLAS library:
    An OpenCL version of BLAS library with default configuration.
  ### 5) Tuned CLBlast BLAS library:
    An OpenCL version of BLAS library tuned for Adreno 540 GPU.  

## 3. Tensorflow MatMul FP32 results:
#### **Square Matrix Multiplication, measurement unit: microsecond (us)
| Library\Size          | OpenCL Kernel Name *[1]     |  N=16   |  N=32   |  N=64   |  N=128  |  N=256  |  N=512  |  N=1024 |  N=2048 |
| :---                  | :---                        | :---    | :---    | :---    | :---    | :---    | :---    | :---    | :---    |
| cpu-Eigen-fp32        | N/A                         | 13      | 29      | 98      | 558     | 8489    | 49335   | 283635  | 3.64e+06|
| tf-kernel 1-fp32      | MatrixMatrixMulSimple       | 9962    | 9744    | 10125   | 11141   | 20353   | 105416  | 802155  | 1.03E+07|
| tf-kernel 2-fp32      | MatrixMatrixMulOptimized2D  | 9460    | 10291   | 9557    | 11180   | 17891   | 68297   | 438467  | 3.67E+06|
| tf-kernel 3-fp32      | MatrixMatrixMulOptimized    | 9703    | 9788    | 9980    | 11826   | 20657   | 68850   | 540829  | 6.84E+06|
| tf-Blast-fp32         | N/A                         | 51555   | 53556   | 53142   | 53525   | 58421   | 85502   | 913113  | 7.01e+06|
| tf-Blast-fp32-tuned   | N/A                         | 118644  | 116903  | 119384  | 117911  | 123486  | 143377  | 165560  | 433448  |
| cl-Blast-fp32         | N/A                         | 1550    | 1890    | 2768    | 2969    | 5395    | 29013   | 860412  | 6719718 |
| cl-Blast-fp32-tuned   | N/A                         | 2467    | 2544    | 2771    | 3209    | 4648    | 20057   | 52937   | 318077  |
##### *[1] See ../opencl-compiler/kernels/GEMM.c for OpenCL Kernel Name
## 4. Testing environment:
Compile ../opencl-matmul project

Run command `./opencl-matmul N 10` where N is the square matrices size
