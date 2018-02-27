# CLBlast SGEMM & HGEMM performance on Adreno 540 [Performance mode]

## 1. xGEMM: General matrix-matrix multiplication
  Performs the matrix product C = alpha * A * B + beta * C, in which A (m by k) and B (k by n)
  are two general rectangular input matrices, C (m by n) is the matrix to be updated, and alpha
  and beta are scalar values. The matrices A and/or B can optionally be transposed before
  performing the operation.

#### ** Eigen & OpenCL results are pure matrix multiplication C = A*B without scalar computation.

## 2. FP32 Results:

#### ** Single Precision FP32, measurement unit: microsecond (us)

| Library\Size      |  N=16   |  N=32   |  N=64   |  N=128  |  N=256  |  N=512  |  N=1024 |  N=2048 |
| :---              | :---    | :---    | :---    | :---    | :---    | :---    | :---    | :---    |
| Eigen             | 13      | 29      | 98      | 558     | 8489    | 49335   | 283635  | 2.35E+06|
| tf-kernel1        | 2536    | 2578    | 2572    |   3972  |  13370  |  100843 | 791522  | 1.13E+07|
| tf-kernel2        | 2590    | 2566    | 2690    | 3032    | 11309   | 64300   | 418989  | 2.06E+06|
| CLBlast [Default] | 52855   | 1890    | 2768    | 2969    | 5395    | 29013   | 860412  | 6719718 |
| CLBlast [Tuned]   | 116440  | 2466    | 2385    | 2857    | 2702    | 3096    | 3230    | 2578    |

## 3. FP16 Results:

#### ** Half Precision FP16, measurement unit: microsecond (us)

| Library\Size      |  N=16   |  N=32   |  N=64   |  N=128  |  N=256  |  N=512  |  N=1024 |  N=2048 |
| :---              | :---    | :---    | :---    | :---    | :---    | :---    | :---    | :---    |
| CLBlast [Default] | 41999   | 2296    | 2440    | 3426    | 4038    | 8489    | 40526   | 218178  |
| CLBlast [Tuned]   | 113213  | 3092    | 2616    | 3426    | 4010    | 8548    | 38593   | 201228  |
