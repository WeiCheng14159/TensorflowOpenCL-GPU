// Matrix Multiplication naive implementation
/*
This kernel computes C = A * B where * means matrix multiplication
A has size (M, K)
B has size (K, N)
C has size (M, N)
*/
__kernel void GEMM(const int M, const int N, const int K,
                      const __global float* A,
                      const __global float* B,
                      __global float* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    //printf("row %d, col %d \n", globalRow, globalCol);
    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k + K*globalRow] * B[globalCol + k*N];
        //printf("A[%d] = %f\n", (k + K*globalRow), A[k + K*globalRow]);
        //printf("B[%d] = %f\n", (globalCol + k*N), B[globalCol + k*N]);
    }

    // Store the result
    C[globalCol + globalRow*N] = acc;
}
