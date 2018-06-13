/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2016                                                     */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern "C" {
  #include <cblas.h>
}

#include "main.h"
#include "init.h"
#include "gpu-op.h"


/*-------------------------------------------------------------------------------*/
/* Global variable declarations.                                                 */
/*-------------------------------------------------------------------------------*/

/* Matrixes: C = A.B                                                             */
/* We use the Transposed B matrix, in place of B, to improve cache memory usage. */
T_real A[SIZE][SIZE];                            /* A Matrix.                    */
T_real B[SIZE][SIZE];                            /* B Matrix.                    */
T_real TB[SIZE][SIZE];                           /* Transposed B Matrix.         */
T_real C[SIZE][SIZE];                            /* C matrix (result matrix).    */

/* Global variables to control OpenMP computations.                              */
int NbThreads = -1;

/* Global vars to control computation on the GPU.                                */
int OnGPUFlag;
ckid_t CPUKernelId;
gkid_t GPUKernelId;


/*-------------------------------------------------------------------------------*/
/* Parallel computation: local computations and data circulations.               */
/*-------------------------------------------------------------------------------*/
void Computation()
{
 // Run computation on the GPU on each node
 if (OnGPUFlag) {
   gpuSetDataOnGPU();
   gpuProduct(GPUKernelId);
   gpuGetResultOnCPU();

 // OR run the computation on the CPU on each node
 } else {
   cpuProduct(CPUKernelId);
 }
}


/*-------------------------------------------------------------------------------*/
/* Local matrix product: optimized code!                                         */
/*-------------------------------------------------------------------------------*/
void cpuProduct(ckid_t kid)
{
 int i, j, k;            // Computation loop indexes
 
 switch(kid) {

 case CK0 : 
   #pragma omp parallel for private(i,j,k)
   for (i = 0; i < SIZE; i++) {
     for (j = 0; j < SIZE; j++) {
       T_real accu[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
       for (k = 0; k < (SIZE/8)*8; k += 8) {
          accu[0] += A[i][k+0] * TB[j][k+0];
          accu[1] += A[i][k+1] * TB[j][k+1];
          accu[2] += A[i][k+2] * TB[j][k+2];
          accu[3] += A[i][k+3] * TB[j][k+3];
          accu[4] += A[i][k+4] * TB[j][k+4];
          accu[5] += A[i][k+5] * TB[j][k+5];
          accu[6] += A[i][k+6] * TB[j][k+6];
          accu[7] += A[i][k+7] * TB[j][k+7];
       }
       for (k = (SIZE/8)*8; k < SIZE; k++) {
          accu[0] += A[i][k] * TB[j][k];
       } 
       C[i][j] = accu[0] + accu[1] + accu[2] + accu[3] + 
                 accu[4] + accu[5] + accu[6] + accu[7];
     }
   }
   break;

 case CK1 : 
   // BLAS kernel
   #pragma omp parallel
   {
     int reste = SIZE % omp_get_num_threads();
     int quotient = SIZE / omp_get_num_threads();
     int NbLig = quotient + 
                 (omp_get_thread_num() < reste ? 1 : 0);
     int offsetLig = quotient*omp_get_thread_num() + 
                     (omp_get_thread_num() < reste ? omp_get_thread_num() : reste);
     CBLAS_GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NbLig, SIZE, SIZE,
                1.0, &A[offsetLig][0], SIZE, 
                &B[0][0], SIZE,
                0.0, &C[offsetLig][0], SIZE);
   }
   break;

 default :
   fprintf(stderr,"Unknown CPU kernel!");
   exit(EXIT_FAILURE);
   break;

 } 
}


/*-------------------------------------------------------------------------------*/
/* Toplevel function.                                                            */
/*-------------------------------------------------------------------------------*/
int main(int argc, char *argv[])
{
    double td1, tf1;                    /* Time measures of the computation loop */
    double td2, tf2;                    /* Time measures of the entire programe  */
    double d1, d2;                      /* Elapsed times to measure.             */
    double gigaflops;                   /* Program performances to measure.      */

    /* Initialisations --------------------------------------------------------- */
    td2 = omp_get_wtime();                        /* Start app. time measurement.*/
    CommandLineParsing(argc,argv);                /* Cmd line parsing.           */
    LocalMatrixInit();                            /* Initialization of the data  */
    omp_set_num_threads(NbThreads);               /* Max nb of threads/node.     */
    if (OnGPUFlag)                                /* Init the GPU device.        */
      gpuInit();

    /* Matrix product computation ---------------------------------------------- */ 
    fprintf(stdout,"Product of two square matrixes of %dx%d T_reals %s:\n",
            SIZE,SIZE,(OnGPUFlag ? "on GPU" : "on CPU"));
    if (OnGPUFlag) {
      fprintf(stdout,"- GPU kernel Id: %d\n", GPUKernelId);
    } else {
      fprintf(stdout,"- CPU kernel Id: %d\n", CPUKernelId);
      fprintf(stdout,"- Max number of OpenMP threads per process: %d\n", NbThreads);
    }
    fprintf(stdout,"- Parallel computation starts...\n");

    td1 = omp_get_wtime();                       
    Computation();               /* TO COMPLETE *//* Parallel Matrix product.    */
    tf1 = omp_get_wtime();                       /* - end of comp. time measure.*/
    tf2 = omp_get_wtime();                       /* - end of app. time measure. */

    /* Performance computation, results and performance printing --------------- */
    d1 = tf1 - td1;                               /* Elapsed comp. time.         */
    d2 = tf2 - td2;                               /* Elapsed app. time.          */
    gigaflops = (2.0*pow(SIZE,3))/d1*1E-9;        /* Performance achieved.       */
    PrintResultsAndPerf(gigaflops,d1,d2);         /* Results and perf printing   */

    if (OnGPUFlag)                                /* Finalize GPU device usage.  */
      gpuFinalize();
      
    /* End of the parallel program --------------------------------------------- */
    return(EXIT_SUCCESS);
}

