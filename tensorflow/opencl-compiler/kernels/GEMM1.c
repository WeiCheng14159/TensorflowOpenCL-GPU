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
    // Each work item is mapped to one thread
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    float acc = 0.0f;
    for (int k=0; k<K; k++) {
        acc += A[k + K*globalRow] * B[globalCol + k*N];
    }

    // Store the result
    C[globalCol + globalRow*N] = acc;
}

// Analysis:
// Each work item is mapped into one element in matrix C
// Memeory access per work item:
// 1) K x 4 bytes of global memeory load from A (coalesced memory access for A)
// 2) K x 4 bytes of global memory load from B (non-coalesced memory access for B)
// 3) 4 bytes of global memory store to C (coalesced memory access for C)
// In total, we use M*N*K*2 memeory load, and M*N memory store, and M*N*K multiplaction.
// The computational intensity is 0.5
