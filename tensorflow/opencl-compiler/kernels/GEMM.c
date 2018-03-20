//--------------------------------------------------------------------------------------
// File: MatrixMatrixMul.cl
// Desc: Matrix-matrix product kernels
//
// Author:      QUALCOMM, Advanced Content Group - Snapdragon SDK
//
//               Copyright (c) 2013-2014 QUALCOMM Technologies, Inc.
//                         All Rights Reserved.
//                      QUALCOMM Proprietary/GTDR
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
// Name: MatrixMatrixMulSimple()
// Desc: Compute the multiplication of two matrices.  Simple approach where each
// work-item computes one cell in the resulting matrix
//--------------------------------------------------------------------------------------
__attribute__((vec_type_hint(float4)))
__attribute__((reqd_work_group_size(16, 16, 0)))
__kernel void MatrixMatrixMulSimple(const int matrixRowsA,
                                    const int matrixColsARowsB,
                                    const int matrixColsB,
                                    __global float* matrixA,
                                    __global float* matrixB,
                                    __global float* matrixProduct)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if( i < matrixRowsA && j < matrixColsB )
    {
        float result = 0.0;
        for( int k = 0; k < matrixColsARowsB; k++ )
        {
            int indexA = i * matrixColsARowsB + k;
            int indexB = k * matrixColsB + j;
            result += matrixA[indexA] * matrixB[indexB];
        }

        matrixProduct[i * matrixColsB + j] = result;
    }
}

//--------------------------------------------------------------------------------------
// Name: MatrixMatrixMulOptimized2D()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a cell of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items.
//--------------------------------------------------------------------------------------
#define LOCAL_MEM_SIZE 16
__attribute__((vec_type_hint(float4)))
__attribute__((reqd_work_group_size(LOCAL_MEM_SIZE, LOCAL_MEM_SIZE, 0)))
__kernel void MatrixMatrixMulOptimized2D( const int matrixRowsA,
                                          const int matrixColsARowsB,
                                          const int matrixColsB,
                                          const __global float* matrixA,
                                          const __global float* matrixB,
                                          __global float* matrixProduct )
{

    // Thread identifiers
    const int localRow = get_local_id(0); // Local row ID (0 .. LOCAL_MEM_SIZE - 1)
    const int localCol = get_local_id(1); // Local col ID (0 . .LOCAL_MEM_SIZE - 1)
    const int globalRow = get_global_id(0); // global row ID of matrixProduct (0 .. matrixRowsA - 1)
    const int globalCol = get_global_id(1); // global col ID of matrixProduct (0 .. matrixColsB - 1)

    // Local memory to fit a tile of LOCAL_MEM_SIZE*LOCAL_MEM_SIZE elements of A and B
    __local float Asub[LOCAL_MEM_SIZE][LOCAL_MEM_SIZE];
    __local float Bsub[LOCAL_MEM_SIZE][LOCAL_MEM_SIZE];

    // Initialise the accumulation register
    float acc = 0.0f;

    //Loop over all tiles
    const int numTiles = matrixColsARowsB / LOCAL_MEM_SIZE;
    for (int t=0; t<numTiles; t++) {

        // Load one tile of A and B into local memory
        const int tiledRow = LOCAL_MEM_SIZE * t + localRow;
        const int tiledCol = LOCAL_MEM_SIZE * t + localCol;

        Asub[localRow][localCol] = matrixA[tiledCol + globalRow * matrixColsARowsB];
        Bsub[localRow][localCol] = matrixB[globalCol + tiledRow * matrixColsB];

        // Synchronise to make sure the tile is loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Perform the computation for a single tile
        for (int k=0; k<LOCAL_MEM_SIZE ; k++) {
          acc += Asub[localRow][k] * Bsub[k][localCol];
        }

        // Synchronise before loading the next tile
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Store the final result in C
    matrixProduct[globalCol + globalRow * matrixColsB] = acc;

}

//--------------------------------------------------------------------------------------
// Name: MatrixTranspose()
// Desc: Compute the transpose of a matrix
// Tranposing Matrix B (for computing product of A and B) helps avoid column accesses
// for Matrix B which in turn helps take advantage of data locality
//--------------------------------------------------------------------------------------
__kernel
__attribute__((vec_type_hint(float4)))
__attribute__((reqd_work_group_size(32, 0, 0)))
void MatrixTranspose(const int rows,
                              const int cols,
                              __global float* matrix,
                              __global float* matrixTranspose)
{
    int gid = get_global_id(0);
    int indexSrc = cols*gid;
    int iters = cols >> 2;
    int offset = 0;

    for(int i=0; i < iters; i++)
    {
        // Vectorization helps utilize the memory bandwidth better
        float4 tmp1 = vload4(0, &matrix[indexSrc]);

        matrixTranspose[gid+offset] = tmp1.x;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.y;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.z;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.w;
        offset += rows;

        indexSrc += 4;
    }

    for( int i = 0 ; i < fmod(cols, 4.0f) ; i++ ){
        float tmp2 = (*((__global float*)&matrix[indexSrc+i]));

        matrixTranspose[gid+offset] = tmp2;
        offset += rows;
    }

}

//--------------------------------------------------------------------------------------
// Name: MatrixMatrixMulOptimized()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a full row of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items. Also, vectorized
// loads are used to utilize the memory bandwidth better.
//--------------------------------------------------------------------------------------
__kernel
__attribute__((vec_type_hint(float4)))
__attribute__((reqd_work_group_size(16, 0, 0)))
void MatrixMatrixMulOptimized(const int matrixRowsA,
                                        const int matrixColsARowsB,
                                        const int matrixColsB,
                                        __global float* matrixA,
                                        __global float* matrixBTranspose,
                                        __global float* matrixProduct,
                     __local float* dataCacheB)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int resultIndex = gid*matrixColsB;
    int iters = matrixColsARowsB >> 2;

    for(int j=0; j < matrixColsB; j++)
    {
        // Use Local Memory to cache BTranspose's rows
        // Fill in the portion of BTranspose's row that this work-item is responsible for
        int offset = j*matrixColsARowsB;
        for(int k=lid; (((k&3)==0) && k<matrixColsARowsB); k+=lsize)
        {
            *((__local float4*)&dataCacheB[k]) = *((__global float4*)&matrixBTranspose[k+offset]);
        }

        // Insert a barrier so all work-items in the workgroup wait until dataCacheB is filled
        barrier( CLK_LOCAL_MEM_FENCE );

        int indexA = matrixColsARowsB*gid;
        int indexBTranspose = 0;
        float result = 0.0f;
        for(int i=0; i < iters; i++)
        {
            // Vectorization of loads help utilize the memory bandwidth better
            float4 tmpRow = vload4(0, &matrixA[indexA] );
            float4 tmpCol = vload4(0, &dataCacheB[indexBTranspose] );
            result += dot(tmpRow, tmpCol);
            indexBTranspose += 4;
            indexA += 4;
        }

        // Iterate through the remaining part
        for( int i = 0 ; i < fmod(matrixColsARowsB, 4.0f) ; i++ ){
            float tmpRow = matrixA[indexA + i];
            float tmpCol = dataCacheB[indexBTranspose + i];
            result += tmpRow * tmpCol;
        }

        matrixProduct[resultIndex+j] = result;

        // This barrier makes sure all reads from dataCacheB complete before the next iteration
        // where the data will be written to again
        barrier( CLK_LOCAL_MEM_FENCE );
    }
}
