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

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define dot8(a,b) \
    (dot(a.hi, b.hi) + dot(a.lo, b.lo))

#define dot16(a,b) \
    (dot8(a.hi, b.hi) + dot8(a.lo, b.lo))

//=============================================================================//
//                                  2D kernel                                  //
//=============================================================================//

//--------------------------------------------------------------------------------------
// Name: MatMul_NN_2D_Simple_Fp32()
// Desc: Compute the multiplication of two matrices.  Simple approach where each
// work-item computes one cell in the resulting matrix
//--------------------------------------------------------------------------------------
__attribute__((reqd_work_group_size(16, 16, 0)))
__kernel void MatMul_NN_2D_Simple_Fp32(
                                    const ushort matrixRowsA,
                                    const ushort matrixColsARowsB,
                                    const ushort matrixColsB,
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
// Name: MatMul_NN_2D_LocalMem_Fp32()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a cell of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items.
//--------------------------------------------------------------------------------------
#define LOCAL_MEM_SIZE 16
__attribute__((reqd_work_group_size(LOCAL_MEM_SIZE, LOCAL_MEM_SIZE, 0)))
__kernel void MatMul_NN_2D_LocalMem_Fp32(
                                      const ushort matrixRowsA,
                                      const ushort matrixColsARowsB,
                                      const ushort matrixColsB,
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

//=============================================================================//
//                                  1D kernel                                  //
//=============================================================================//

//=============================================================================//
//                         Mat Transpose kernel (FP32)                         //
//=============================================================================//

//--------------------------------------------------------------------------------------
// Name: MatTrans_1D_Fp32_Float4()
// Desc: Compute the transpose of a matrix
// Tranposing Matrix B (for computing product of A and B) helps avoid column accesses
// for Matrix B which in turn helps take advantage of data locality
//--------------------------------------------------------------------------------------
__kernel
void MatTrans_1D_Fp32_Float4(
                                const ushort rows,
                                const ushort cols,
                                __global float* matrix,
                                __global float* matrixTranspose)
{
    ushort gid = get_global_id(0);
    int indexSrc = mul24(cols, gid);
    ushort iters = cols >> 2;
    int offset = 0;

    for(ushort i=0; i < iters; i++)
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

    for( ushort i = 0 ; i < (cols & 3) ; i++ ){
        float tmp2 = (*((__global float*)&matrix[indexSrc+i]));

        matrixTranspose[gid+offset] = tmp2;
        offset += rows;
    }

}

//--------------------------------------------------------------------------------------
// Name: MatTrans_1D_Fp32_Float8()
// Desc: Compute the transpose of a matrix
// Tranposing Matrix B (for computing product of A and B) helps avoid column accesses
// for Matrix B which in turn helps take advantage of data locality
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(32, 0, 0)))
void MatTrans_1D_Fp32_Float8(
                                const ushort rows,
                                const ushort cols,
                                __global float* matrix,
                                __global float* matrixTranspose)
{
    ushort gid = get_global_id(0);
    int indexSrc = mul24(cols, gid);
    ushort iters = cols >> 3;
    int offset = 0;

    for(ushort i=0; i < iters; i++)
    {
        // Vectorization helps utilize the memory bandwidth better
        float8 tmp1 = vload8(0, &matrix[indexSrc]);

        matrixTranspose[gid+offset] = tmp1.s0;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.s1;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.s2;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.s3;
        offset += rows;

        matrixTranspose[gid+offset] = tmp1.s4;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.s5;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.s6;
        offset += rows;
        matrixTranspose[gid+offset] = tmp1.s7;
        offset += rows;

        indexSrc += 8;
    }

    for( ushort i = 0 ; i < (cols & 7) ; i++ ){
        float tmp2 = (*((__global float*)&matrix[indexSrc+i]));

        matrixTranspose[gid+offset] = tmp2;
        offset += rows;
    }

}

//--------------------------------------------------------------------------------------
// Name: MatTrans_1D_Fp32_Float16()
// Desc: Compute the transpose of a matrix
// Tranposing Matrix B (for computing product of A and B) helps avoid column accesses
// for Matrix B which in turn helps take advantage of data locality
//--------------------------------------------------------------------------------------
__kernel
void MatTrans_1D_Fp32_Float16(
                                const ushort rows,
                                const ushort cols,
                                __global float* matrix,
                                __global float* matrixTranspose)
{
    ushort gid = get_global_id(0);
    int indexSrc = mul24(cols, gid);
    ushort iters = cols >> 4;
    int offset = 0;

    for(ushort i=0; i < iters; i++)
    {
        // Vectorization helps utilize the memory bandwidth better
        float8 float8Vec_1 = vload8(0, &matrix[indexSrc]);
        float8 float8Vec_2 = vload8(0, &matrix[indexSrc+8]);

        // float8Vec_1
        matrixTranspose[gid+offset] = float8Vec_1.s0;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_1.s1;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_1.s2;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_1.s3;
        offset += rows;

        matrixTranspose[gid+offset] = float8Vec_1.s4;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_1.s5;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_1.s6;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_1.s7;
        offset += rows;

        // float8Vec_2
        matrixTranspose[gid+offset] = float8Vec_2.s0;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_2.s1;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_2.s2;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_2.s3;
        offset += rows;

        matrixTranspose[gid+offset] = float8Vec_2.s4;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_2.s5;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_2.s6;
        offset += rows;
        matrixTranspose[gid+offset] = float8Vec_2.s7;
        offset += rows;

        indexSrc += 16;
    }

    for( ushort i = 0 ; i < (cols & 15) ; i++ ){
        float tmp2 = (*((__global float*)&matrix[indexSrc+i]));

        matrixTranspose[gid+offset] = tmp2;
        offset += rows;
    }

}

//=============================================================================//
//                         Mat Transpose kernel (FP16)                         //
//=============================================================================//

//--------------------------------------------------------------------------------------
// Name: MatTrans_1D_Fp16_Half4()
// Desc: Compute the transpose of a matrix
// Tranposing Matrix B (for computing product of A and B) helps avoid column accesses
// for Matrix B which in turn helps take advantage of data locality
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(16, 0, 0)))
void MatTrans_1D_Fp16_Half4(
                                const ushort rows,
                                const ushort cols,
                                __global half* matrix,
                                __global half* matrixTranspose)
{
    ushort gid = get_global_id(0);
    int indexSrc = mul24(cols, gid);
    ushort iters = cols >> 2;
    int offset = 0;

    for(ushort i=0; i < iters; i++)
    {
        // Vectorization helps utilize the memory bandwidth better
        half4 Half4Vec = vload4(0, &matrix[indexSrc]);

        matrixTranspose[gid+offset] = Half4Vec.x;
        offset += rows;
        matrixTranspose[gid+offset] = Half4Vec.y;
        offset += rows;
        matrixTranspose[gid+offset] = Half4Vec.z;
        offset += rows;
        matrixTranspose[gid+offset] = Half4Vec.w;
        offset += rows;

        indexSrc += 4;
    }

    for( int i = 0 ; i < (cols & 3) ; i++ ){
        half tmp2 = (*((__global half*)&matrix[indexSrc+i]));

        matrixTranspose[gid+offset] = tmp2;
        offset += rows;
    }

}

//--------------------------------------------------------------------------------------
// Name: MatTrans_1D_Fp16_Half8()
// Desc: Compute the transpose of a matrix
// Tranposing Matrix B (for computing product of A and B) helps avoid column accesses
// for Matrix B which in turn helps take advantage of data locality
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(16, 0, 0)))
void MatTrans_1D_Fp16_Half8(
                                const ushort rows,
                                const ushort cols,
                                __global half* matrix,
                                __global half* matrixTranspose)
{
    ushort gid = get_global_id(0);
    int indexSrc = mul24(cols, gid);
    ushort iters = cols >> 3;
    int offset = 0;

    for(ushort i=0; i < iters; i++)
    {
        // Vectorization helps utilize the memory bandwidth better
        half8 Half8Vec = vload8(0, &matrix[indexSrc]);

        matrixTranspose[gid+offset] = Half8Vec.s0;
        offset += rows;
        matrixTranspose[gid+offset] = Half8Vec.s1;
        offset += rows;
        matrixTranspose[gid+offset] = Half8Vec.s2;
        offset += rows;
        matrixTranspose[gid+offset] = Half8Vec.s3;
        offset += rows;

        matrixTranspose[gid+offset] = Half8Vec.s4;
        offset += rows;
        matrixTranspose[gid+offset] = Half8Vec.s5;
        offset += rows;
        matrixTranspose[gid+offset] = Half8Vec.s6;
        offset += rows;
        matrixTranspose[gid+offset] = Half8Vec.s7;
        offset += rows;

        indexSrc += 8;
    }

    for( int i = 0 ; i < (cols & 7) ; i++ ){
        half tmp2 = (*((__global half*)&matrix[indexSrc+i]));

        matrixTranspose[gid+offset] = tmp2;
        offset += rows;
    }

}

//--------------------------------------------------------------------------------------
// Name: MatTrans_1D_Fp16_Half16()
// Desc: Compute the transpose of a matrix
// Tranposing Matrix B (for computing product of A and B) helps avoid column accesses
// for Matrix B which in turn helps take advantage of data locality
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(16, 0, 0)))
void MatTrans_1D_Fp16_Half16(
                                const ushort rows,
                                const ushort cols,
                                __global half* matrix,
                                __global half* matrixTranspose)
{
    ushort gid = get_global_id(0);
    int indexSrc = mul24(cols, gid);
    ushort iters = cols >> 4;
    int offset = 0;

    for(ushort i=0; i < iters; i++)
    {
        // Vectorization helps utilize the memory bandwidth better
        half8 Half16Vec_1 = vload8(0, &matrix[indexSrc]);
        half8 Half16Vec_2 = vload8(0, &matrix[indexSrc+8]);

        // Half16Vec_1
        matrixTranspose[gid+offset] = Half16Vec_1.s0;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_1.s1;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_1.s2;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_1.s3;
        offset += rows;

        matrixTranspose[gid+offset] = Half16Vec_1.s4;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_1.s5;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_1.s6;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_1.s7;
        offset += rows;

        // Half16Vec_2
        matrixTranspose[gid+offset] = Half16Vec_2.s0;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_2.s1;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_2.s2;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_2.s3;
        offset += rows;

        matrixTranspose[gid+offset] = Half16Vec_2.s4;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_2.s5;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_2.s6;
        offset += rows;
        matrixTranspose[gid+offset] = Half16Vec_2.s7;
        offset += rows;

        indexSrc += 16;
    }

    for( int i = 0 ; i < (cols & 15) ; i++ ){
        half tmp2 = (*((__global half*)&matrix[indexSrc+i]));

        matrixTranspose[gid+offset] = tmp2;
        offset += rows;
    }

}

//=============================================================================//
//                           Mat Mul kernel (FP32)                             //
//=============================================================================//

//--------------------------------------------------------------------------------------
// Name: MatMul_TN_1D_Fp32_Float4()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a full row of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items. Also, vectorized
// loads are used to utilize the memory bandwidth better.
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(16, 0, 0))) // Best WG size
void MatMul_TN_1D_Fp32_Float4(
                                const ushort matrixRowsA,
                                const ushort matrixColsARowsB,
                                const ushort matrixColsB,
                                __global float* matrixA,
                                __global float* matrixBTranspose,
                                __global float* matrixProduct,
                                __local float* dataCacheB)
{
    ushort gid = get_global_id(0);
    ushort lid = get_local_id(0);
    ushort lsize = get_local_size(0);
    int resultIndex = mul24(gid, matrixColsB);
    ushort iters = matrixColsARowsB >> 2;

    for(ushort j=0; j < matrixColsB; j++)
    {
        // Use Local Memory to cache BTranspose's rows
        // Fill in the portion of BTranspose's row that this work-item is responsible for
        int offset = j*matrixColsARowsB;
        for(ushort k=lid; k<matrixColsARowsB; k+=lsize)
        {
          if( (k & 3) == 0 ){ // k% 4 == 0
            *((__local float4*)&dataCacheB[k]) = *((__global float4*)&matrixBTranspose[k+offset]);
          }
        }

        // Insert a barrier so all work-items in the workgroup wait until dataCacheB is filled
        barrier( CLK_LOCAL_MEM_FENCE );

        int indexA = matrixColsARowsB*gid;
        ushort indexBTranspose = 0;
        float result = 0.0f;
        for(ushort i=0; i < iters; i++)
        {
            // Vectorization of loads help utilize the memory bandwidth better
            float4 float4RowVec = vload4(0, &matrixA[indexA] );
            float4 float4ColVec = vload4(0, &dataCacheB[indexBTranspose] );
            result += dot(float4RowVec, float4ColVec);
            indexBTranspose += 4;
            indexA += 4;
        }

        // Iterate through the remaining part
        for( ushort i = 0 ; i < ( matrixColsARowsB & 3) ; i++ ){
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

//--------------------------------------------------------------------------------------
// Name: MatMul_TN_1D_Fp32_Float8()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a full row of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items. Also, vectorized
// loads are used to utilize the memory bandwidth better.
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(16, 0, 0))) // Best WG size
void MatMul_TN_1D_Fp32_Float8(
                                const ushort matrixRowsA,
                                const ushort matrixColsARowsB,
                                const ushort matrixColsB,
                                __global float* matrixA,
                                __global float* matrixBTranspose,
                                __global float* matrixProduct,
                                __local float* dataCacheB)
{
    ushort gid = get_global_id(0);
    ushort lid = get_local_id(0);
    ushort lsize = get_local_size(0);
    int resultIndex = mul24(gid, matrixColsB);
    ushort iters = matrixColsARowsB >> 3;

    for(ushort j=0; j < matrixColsB; j++)
    {
        // Use Local Memory to cache BTranspose's rows
        // Fill in the portion of BTranspose's row that this work-item is responsible for
        int offset = j*matrixColsARowsB;
        for(ushort k=lid; k<matrixColsARowsB; k+=lsize)
        {
          if( (k & 7) == 0 ){ // k% 8 == 0
            *((__local float8*)&dataCacheB[k]) = *((__global float8*)&matrixBTranspose[k+offset]);
          }
        }

        // Insert a barrier so all work-items in the workgroup wait until dataCacheB is filled
        barrier( CLK_LOCAL_MEM_FENCE );

        int indexA = matrixColsARowsB*gid;
        ushort indexBTranspose = 0;
        float result = 0.0f;
        for(ushort i=0; i < iters; i++)
        {
            // Vectorization of loads help utilize the memory bandwidth better
            float8 float8RowVec = vload8(0, &matrixA[indexA] );
            float8 float8ColVec = vload8(0, &dataCacheB[indexBTranspose] );
            result += dot8(float8RowVec, float8ColVec);
            indexBTranspose += 8;
            indexA += 8;
        }

        // Iterate through the remaining part
        for( ushort i = 0 ; i < ( matrixColsARowsB & 7) ; i++ ){
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

//--------------------------------------------------------------------------------------
// Name: MatMul_TN_1D_Fp32_Float16()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a full row of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items. Also, vectorized
// loads are used to utilize the memory bandwidth better.
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(64, 0, 0))) // Best WG size
void MatMul_TN_1D_Fp32_Float16(
                                const ushort matrixRowsA,
                                const ushort matrixColsARowsB,
                                const ushort matrixColsB,
                                __global float* matrixA,
                                __global float* matrixBTranspose,
                                __global float* matrixProduct,
                                __local float* dataCacheB)
{
    ushort gid = get_global_id(0);
    ushort lid = get_local_id(0);
    ushort lsize = get_local_size(0);
    int resultIndex = mul24(gid, matrixColsB);
    ushort iters = matrixColsARowsB >> 4;

    for(ushort j=0; j < matrixColsB; j++)
    {
        // Use Local Memory to cache BTranspose's rows
        // Fill in the portion of BTranspose's row that this work-item is responsible for
        int offset = j*matrixColsARowsB;
        for(ushort k=lid; k<matrixColsARowsB; k+=lsize)
        {
          if( (k & 15) == 0 ){ // k % 16 == 0
            *((__local float16*)&dataCacheB[k]) = *((__global float16*)&matrixBTranspose[k+offset]);
          }
        }

        // Insert a barrier so all work-items in the workgroup wait until dataCacheB is filled
        barrier( CLK_LOCAL_MEM_FENCE );

        int indexA = matrixColsARowsB*gid;
        ushort indexBTranspose = 0;
        float result = 0.0f;
        for(ushort i=0; i < iters; i++)
        {
            // Vectorization of loads help utilize the memory bandwidth better
            float16 float16RowVec = vload16(0, &matrixA[indexA] );
            float16 float16ColVec = vload16(0, &dataCacheB[indexBTranspose] );
            result += dot16(float8RowVec, float8ColVec);
            indexBTranspose += 16;
            indexA += 16;
        }

        // Iterate through the remaining part
        for( ushort i = 0 ; i < ( matrixColsARowsB & 15) ; i++ ){
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

//=============================================================================//
//                           Mat Mul kernel (FP16)                             //
//=============================================================================//

//--------------------------------------------------------------------------------------
// Name: MatMul_TN_1D_Fp16_Half4()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a full row of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items. Also, vectorized
// loads are used to utilize the memory bandwidth better.
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(16, 0, 0)))
void MatMul_TN_1D_Fp16_Half4(
                              const ushort matrixRowsA,
                              const ushort matrixColsARowsB,
                              const ushort matrixColsB,
                              __global half* matrixA,
                              __global half* matrixBTranspose,
                              __global float* matrixProduct,
                              __local  half* dataCacheB)
{
  ushort gid = get_global_id(0);
  ushort lid = get_local_id(0);
  ushort lsize = get_local_size(0);
  int resultIndex = mul24(gid, matrixColsB);
  ushort iters = matrixColsARowsB >> 2;

  for(ushort j=0; j < matrixColsB; j++)
  {
      // Use Local Memory to cache BTranspose's rows
      // Fill in the portion of BTranspose's row that this work-item is responsible for
      int offset = j*matrixColsARowsB;
      for(ushort k=lid; k<matrixColsARowsB; k+=lsize)
      {
        if( (k & 3) == 0 ){ // k% 4 == 0
          *((__local half4*)&dataCacheB[k]) = *((__global half4*)&matrixBTranspose[k+offset]);
        }
      }

      // Insert a barrier so all work-items in the workgroup wait until dataCacheB is filled
      barrier( CLK_LOCAL_MEM_FENCE );

      int indexA = matrixColsARowsB*gid;
      ushort indexBTranspose = 0;
      float result = 0.0f;
      for(ushort i=0; i < iters; i++)
      {
          // Vectorization of loads help utilize the memory bandwidth better
          float4 float4RowVec = vload_half4(0, &matrixA[indexA] );
          float4 float4ColVec = vload_half4(0, &dataCacheB[indexBTranspose] );
          result += dot(float4RowVec, float4ColVec);
          indexBTranspose += 4;
          indexA += 4;
      }

      // Iterate through the remaining part
      for( ushort i = 0 ; i < ( matrixColsARowsB & 3) ; i++ ){
          float tmpRow = vload_half(0, &matrixA[indexA + i]);
          float tmpCol = vload_half(0, &dataCacheB[indexBTranspose + i]);
          result += tmpRow * tmpCol;
      }

      matrixProduct[resultIndex+j] = result;

      // This barrier makes sure all reads from dataCacheB complete before the next iteration
      // where the data will be written to again
      barrier( CLK_LOCAL_MEM_FENCE );
  }
}

//--------------------------------------------------------------------------------------
// Name: MatMul_TN_1D_Fp16_Half8()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a full row of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items. Also, vectorized
// loads are used to utilize the memory bandwidth better.
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(16, 0, 0)))
void MatMul_TN_1D_Fp16_Half8(
                              const ushort matrixRowsA,
                              const ushort matrixColsARowsB,
                              const ushort matrixColsB,
                              __global half* matrixA,
                              __global half* matrixBTranspose,
                              __global float* matrixProduct,
                              __local  half* dataCacheB)
{
  ushort gid = get_global_id(0);
  ushort lid = get_local_id(0);
  ushort lsize = get_local_size(0);
  int resultIndex = mul24(gid, matrixColsB);
  ushort iters = matrixColsARowsB >> 3;

  for(ushort j=0; j < matrixColsB; j++)
  {
      // Use Local Memory to cache BTranspose's rows
      // Fill in the portion of BTranspose's row that this work-item is responsible for
      int offset = j*matrixColsARowsB;
      for(ushort k=lid; k<matrixColsARowsB; k+=lsize)
      {
        if( (k & 7) == 0 ){ // k % 8 == 0
          *((__local half8*)&dataCacheB[k]) = *((__global half8*)&matrixBTranspose[k+offset]);
        }
      }

      // Insert a barrier so all work-items in the workgroup wait until dataCacheB is filled
      barrier( CLK_LOCAL_MEM_FENCE );

      int indexA = matrixColsARowsB*gid;
      ushort indexBTranspose = 0;
      float result = 0.0f;
      for(ushort i=0; i < iters; i++)
      {
          // Vectorization of loads help utilize the memory bandwidth better
          float8 float8RowVec = vload_half8(0, &matrixA[indexA] );
          float8 float8ColVec = vload_half8(0, &dataCacheB[indexBTranspose] );
          result += dot8(float8RowVec, float8ColVec);
          indexBTranspose += 8;
          indexA += 8;
      }

      // Iterate through the remaining part
      for( ushort i = 0 ; i < ( matrixColsARowsB & 7) ; i++ ){
          float tmpRow = vload_half(0, &matrixA[indexA + i]);
          float tmpCol = vload_half(0, &dataCacheB[indexBTranspose + i]);
          result += tmpRow * tmpCol;
      }

      matrixProduct[resultIndex+j] = result;

      // This barrier makes sure all reads from dataCacheB complete before the next iteration
      // where the data will be written to again
      barrier( CLK_LOCAL_MEM_FENCE );
  }
}

//--------------------------------------------------------------------------------------
// Name: MatMul_TN_1D_Fp16_Half16()
// Desc: Compute the multiplication of two matrices.  In this case, each work-item
// computes a full row of the resulting matrix C.  Additionally, local memory is
// used to cache intermediate fetched values across work-items. Also, vectorized
// loads are used to utilize the memory bandwidth better.
//--------------------------------------------------------------------------------------
__kernel
__attribute__((reqd_work_group_size(16, 0, 0)))
void MatMul_TN_1D_Fp16_Half16(
                              const ushort matrixRowsA,
                              const ushort matrixColsARowsB,
                              const ushort matrixColsB,
                              __global half* matrixA,
                              __global half* matrixBTranspose,
                              __global float* matrixProduct,
                              __local  half* dataCacheB)
{
  ushort gid = get_global_id(0);
  ushort lid = get_local_id(0);
  ushort lsize = get_local_size(0);
  int resultIndex = mul24(gid, matrixColsB);
  ushort iters = matrixColsARowsB >> 4;

  for(ushort j=0; j < matrixColsB; j++)
  {
      // Use Local Memory to cache BTranspose's rows
      // Fill in the portion of BTranspose's row that this work-item is responsible for
      int offset = j*matrixColsARowsB;
      for(ushort k=lid; k<matrixColsARowsB; k+=lsize)
      {
        if( (k & 15) == 0 ){ // k % 16 == 0
          *((__local half16*)&dataCacheB[k]) = *((__global half16*)&matrixBTranspose[k+offset]);
        }
      }

      // Insert a barrier so all work-items in the workgroup wait until dataCacheB is filled
      barrier( CLK_LOCAL_MEM_FENCE );

      int indexA = matrixColsARowsB*gid;
      ushort indexBTranspose = 0;
      float result = 0.0f;
      for(ushort i=0; i < iters; i++)
      {
          // Vectorization of loads help utilize the memory bandwidth better
          float16 float16RowVec = vload_half16(0, &matrixA[indexA] );
          float16 float16ColVec = vload_half16(0, &dataCacheB[indexBTranspose] );
          result += dot16(tmpRow, tmpCol);
          indexBTranspose += 16;
          indexA += 16;
      }

      // Iterate through the remaining part
      for( ushort i = 0 ; i < ( matrixColsARowsB & 15) ; i++ ){
          float tmpRow = vload_half(0, &matrixA[indexA + i]);
          float tmpCol = vload_half(0, &dataCacheB[indexBTranspose + i]);
          result += tmpRow * tmpCol;
      }

      matrixProduct[resultIndex+j] = result;

      // This barrier makes sure all reads from dataCacheB complete before the next iteration
      // where the data will be written to again
      barrier( CLK_LOCAL_MEM_FENCE );
  }
}
