/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2016                                                     */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "main.h"
#include "gpu-op.h"


/*-------------------------------------------------------------------------------*/
/* GPU symbols                                                                   */
/*-------------------------------------------------------------------------------*/
__device__ T_real GPU_A[SIZE][SIZE];
__device__ T_real GPU_B[SIZE][SIZE];
__device__ T_real GPU_C[SIZE][SIZE];


/*-------------------------------------------------------------------------------*/
/* Init and finalize the GPU device.                                             */
/*-------------------------------------------------------------------------------*/
void gpuInit(void)
{
  cuInit(0);
}


void gpuFinalize(void)
{

}


/*-------------------------------------------------------------------------------*/
/* Transfer of CPU input data into GPU symbols                                   */
/*-------------------------------------------------------------------------------*/
void gpuSetDataOnGPU(void)
{
 // Set GPU_A symbol
 // Transfer A-->GPU_A
 CHECK_CUDA_SUCCESS(cudaMemcpyToSymbol(GPU_A, &A, sizeof(T_real) * SIZE * SIZE, 0, cudaMemcpyHostToDevice), "Copying A to GPU");

 // Set GPU_B symbol
 // Transfer B-->GPU_B

 CHECK_CUDA_SUCCESS(cudaMemcpyToSymbol(GPU_B, &B, sizeof(T_real) * SIZE * SIZE, 0, cudaMemcpyHostToDevice), "Copying B to GPU");
}


/*-------------------------------------------------------------------------------*/
/* Transfer of GPU results into CPU array                                        */
/*-------------------------------------------------------------------------------*/
void gpuGetResultOnCPU(void)
{
 // Get GPU_C symbol
 // Transfer GPU C-->C
 CHECK_CUDA_SUCCESS(cudaMemcpyFromSymbol(&C, GPU_C, sizeof(T_real) * SIZE * SIZE, 0, cudaMemcpyDeviceToHost), "Fetching C from GPU");
}


/*-------------------------------------------------------------------------------*/
/* Small matrix product on the local GPU.                                        */
/*-------------------------------------------------------------------------------*/
__global__ void MatrixProductKernel_v0(void)
{
 // Index computations
 int lig = blockIdx.y;
 int col = BLOCK_SIZE_X_K0 * blockIdx.x + threadIdx.x;
 T_real res = 0.0;
 // Matrix product computation
 for(int k = 0; k < SIZE ; k ++ ) {
   res += GPU_A[lig][k] * GPU_B[k][col];
 }
 GPU_C[lig][col] = res;
}

__global__ void MatrixProductKernel_v1(void)
{
 // Index computations
 int lig = blockIdx.y;
 int col = BLOCK_SIZE_X_K0 * blockIdx.x + threadIdx.x;
 if(col < SIZE) {
   T_real res = 0.0;
   // Matrix product computation
   for(int k = 0; k < SIZE ; k ++ ) {
     res += GPU_A[lig][k] * GPU_B[k][col];
   }
   GPU_C[lig][col] = res;
 }
}

__global__ void MatrixProductKernel_v2(void)
{
 // Index computations
 int lig = BLOCK_SIZE_Y_K1 * blockIdx.y + threadIdx.y;
 int col = BLOCK_SIZE_X_K1 * blockIdx.x + threadIdx.x;
 if(col < SIZE &&  lig < SIZE) {
   T_real res = 0.0;
   // Matrix product computation
   for(int k = 0; k < SIZE ; k ++ ) {
     res += GPU_A[lig][k] * GPU_B[k][col];
   }
   GPU_C[lig][col] = res;
 }
}


/*-------------------------------------------------------------------------------*/
/* Small matrix product on the local GPU.                                        */
/*-------------------------------------------------------------------------------*/
void gpuProduct(gkid_t kid)
{
 dim3 Dg, Db;

 switch(kid) {

 case GK0 : // Kernel v0 - using only global memory (with coalescent data accesses)
   // - init the grid of blocs
  Db.x = BLOCK_SIZE_X_K0;
  Db.y = 1;
  Db.z = 1;
  Dg.x = SIZE/BLOCK_SIZE_X_K0;
  Dg.y = SIZE;
  Dg.z = 1;
   // - run the Grid of Blocs of threads
  MatrixProductKernel_v0<<<Dg,Db>>>();
   break;

 case GK1 :
   Db.x = BLOCK_SIZE_X_K0;
   Db.y = 1;
   Db.z = 1;
   Dg.x = SIZE/BLOCK_SIZE_X_K0 + (SIZE % BLOCK_SIZE_X_K0 ? 1 : 0);
   Dg.y = SIZE;
   Dg.z = 1;
    // - run the Grid of Blocs of threads
   MatrixProductKernel_v1<<<Dg,Db>>>();
  break;

 case GK2 :
   Db.x = BLOCK_SIZE_X_K1;
   Db.y = BLOCK_SIZE_Y_K1;
   Db.z = 1;
   Dg.x = SIZE/BLOCK_SIZE_X_K1 + (SIZE % BLOCK_SIZE_X_K1 ? 1 : 0);
   Dg.y = SIZE/BLOCK_SIZE_Y_K1 + (SIZE % BLOCK_SIZE_Y_K1 ? 1 : 0);
   Dg.z = 1;
    // - run the Grid of Blocs of threads
   MatrixProductKernel_v2<<<Dg,Db>>>();
  break;

 case GK3 :
  break;

 case GK4 :
  break;

 case GK5 :
  break;

 default :
   fprintf(stderr,"Unknown GPU kernel!");
   exit(EXIT_FAILURE);
 }
}
