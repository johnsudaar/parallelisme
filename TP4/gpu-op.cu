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

__global__ void MatrixProductKernel_v3(void)
{
  __shared__ T_real SHARED_A[BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];
  __shared__ T_real SHARED_B[BLOCK_SIZE_XY_K2][BLOCK_SIZE_XY_K2];

  T_real accu = 0.0;
 // Index computations
 int lig = BLOCK_SIZE_XY_K2 * blockIdx.y + threadIdx.y;
 int col = BLOCK_SIZE_XY_K2 * blockIdx.x + threadIdx.x;

 for(int step = 0; step < gridDim.x ; step++) {
   int offsetStep = step * BLOCK_SIZE_XY_K2;
   int gcolA = offsetStep + threadIdx.x;
   int gligB = offsetStep + threadIdx.y;
   SHARED_A[threadIdx.y][threadIdx.x] = GPU_A[lig][gcolA];
   SHARED_B[threadIdx.y][threadIdx.x] = GPU_B[gligB][col];

   __syncthreads();
   for(int k = 0; k < BLOCK_SIZE_XY_K2; k++) {
     accu += SHARED_A[threadIdx.y][k] * SHARED_B[k][threadIdx.x];
   }
   __syncthreads();
 }

 GPU_C[lig][col] = accu;
}

__global__ void MatrixProductKernel_v4(void)
{
  __shared__ T_real SHARED_A[BLOCK_SIZE_XY_K3][BLOCK_SIZE_XY_K3];
  __shared__ T_real SHARED_B[BLOCK_SIZE_XY_K3][BLOCK_SIZE_XY_K3];

  T_real accu = 0.0;
 // Index computations
 int lig = BLOCK_SIZE_XY_K3 * blockIdx.y + threadIdx.y;
 int col = BLOCK_SIZE_XY_K3 * blockIdx.x + threadIdx.x;

 for(int step = 0; step < gridDim.x ; step++) {
   int offsetStep = step * BLOCK_SIZE_XY_K3;
   int gcolA = offsetStep + threadIdx.x;
   int gligB = offsetStep + threadIdx.y;
   if(gcolA < SIZE && lig < SIZE) {
     SHARED_A[threadIdx.y][threadIdx.x] = GPU_A[lig][gcolA];
   } else {
     SHARED_A[threadIdx.y][threadIdx.x] = 0;
   }

   if(gligB < SIZE && col < SIZE) {
     SHARED_B[threadIdx.y][threadIdx.x] = GPU_B[gligB][col];
   } else {
     SHARED_B[threadIdx.y][threadIdx.x] = 0;
   }

   __syncthreads();
   for(int k = 0; k < BLOCK_SIZE_XY_K3; k++) {
     accu += SHARED_A[threadIdx.y][k] * SHARED_B[k][threadIdx.x];
   }
   __syncthreads();
 }
 if(lig < SIZE && col < SIZE) {
   GPU_C[lig][col] = accu;
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
   Db.x = BLOCK_SIZE_XY_K2;
   Db.y = BLOCK_SIZE_XY_K2;
   Db.z = 1;
   Dg.x = SIZE/BLOCK_SIZE_XY_K2 + (SIZE % BLOCK_SIZE_XY_K2 ? 1 : 0);
   Dg.y = SIZE/BLOCK_SIZE_XY_K2 + (SIZE % BLOCK_SIZE_XY_K2 ? 1 : 0);
   Dg.z = 1;
    // - run the Grid of Blocs of threads
   MatrixProductKernel_v3<<<Dg,Db>>>();
  break;

 case GK4 :
   Db.x = BLOCK_SIZE_XY_K3;
   Db.y = BLOCK_SIZE_XY_K3;
   Db.z = 1;
   Dg.x = SIZE/BLOCK_SIZE_XY_K3 + (SIZE % BLOCK_SIZE_XY_K3 ? 1 : 0);
   Dg.y = SIZE/BLOCK_SIZE_XY_K3 + (SIZE % BLOCK_SIZE_XY_K3 ? 1 : 0);
   Dg.z = 1;
    // - run the Grid of Blocs of threads
   MatrixProductKernel_v4<<<Dg,Db>>>();
  break;

 case GK5 :
  break;

 default :
   fprintf(stderr,"Unknown GPU kernel!");
   exit(EXIT_FAILURE);
 }
}
