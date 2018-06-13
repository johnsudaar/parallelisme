/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2016                                                     */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "main.h"
#include "init.h"


/*-------------------------------------------------------------------------------*/
/* Initialisation of local matrixes A, B and C                                   */
/* Each process initializes its local parts of matrixes: simulates a parallel    */
/* initialization from files on disks.                                           */
/*-------------------------------------------------------------------------------*/
void LocalMatrixInit(void)
{
 int i, j;                                /* Local matrix indexes                */

/* Initialization of the local matrix elements                                   */
 for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++)
       A[i][j] = (T_real) (i*SIZE + j);
 
 for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++) {
       B[i][j]  = (T_real) (i*SIZE + j);
       TB[j][i] = (T_real) (i*SIZE + j);
    }
 
 for (i = 0; i < SIZE; i++)
    for (j = 0; j < SIZE; j++)
       C[i][j] = 0.0;
}


/*-------------------------------------------------------------------------------*/
/* Command Line parsing.                                                         */
/*-------------------------------------------------------------------------------*/
void usage(int ExitCode, FILE *std)
{
 fprintf(std,"MatrixProduct usage: \n");
 fprintf(std,"\t [-h]: print this help\n");
 fprintf(std,"\t [-c <GPU(default)|CPU>]: run computations on the GPU or on the CPU\n");
 fprintf(std,"\t [-cpu-k <CPU kernel Id [0(default) - %d]>]\n",(NB_OF_CPU_KERNELS-1));
 fprintf(std,"\t [-cpu-nt <number of OpenMP threads> (default %d)]\n",DEFAULT_NB_THREADS);
 fprintf(std,"\t [-gpu-k <GPU kernel Id [0(default) - %d]>]\n",(NB_OF_GPU_KERNELS-1));

 exit(ExitCode);
}


void CommandLineParsing(int argc, char *argv[])
{
 // Default init
 NbThreads = DEFAULT_NB_THREADS;
 OnGPUFlag = DEFAULT_ONGPUFLAG;
 CPUKernelId = DEFAULT_CPUKID;
 GPUKernelId = DEFAULT_GPUKID;

 // Init from the command line
 argc--; argv++;
 while (argc > 0) {
     if (strcmp(argv[0],"-c") == 0) {
       argc--; argv++;
       if (argc > 0) {
         if (strcmp(argv[0],"GPU") == 0) {
           OnGPUFlag = 1;
           argc--; argv++;
         } else if (strcmp(argv[0],"CPU") == 0) {
           OnGPUFlag = 0;
           argc--; argv++;
         } else {
           fprintf(stderr,"Error: unknown computation mode '%s'!\n",argv[0]);
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-cpu-k") == 0) {
       argc--; argv++;
       if (argc > 0) {
         CPUKernelId = (ckid_t) atoi(argv[0]);
         argc--; argv++;
         if (CPUKernelId < 0 || CPUKernelId >= NB_OF_CPU_KERNELS) {
           fprintf(stderr,"Error: CPU kernel Id has to in [0 - %d]!\n",(NB_OF_CPU_KERNELS-1));
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-cpu-nt") == 0) {
       argc--; argv++;
       if (argc > 0) {
         NbThreads = atoi(argv[0]);
         argc--; argv++;
         if (NbThreads <= 0) {
           fprintf(stderr,"Error: number of thread has to be >= 1!\n");
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }

     } else if (strcmp(argv[0],"-gpu-k") == 0) {
       argc--; argv++;
       if (argc > 0) {
         GPUKernelId = (gkid_t) atoi(argv[0]);
         argc--; argv++;
         if (GPUKernelId < 0 || GPUKernelId >= NB_OF_GPU_KERNELS) {
           fprintf(stderr,"Error: GPU kernel Id has to in [0 - %d]!\n",(NB_OF_GPU_KERNELS-1));
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }


     } else if (strcmp(argv[0],"-h") == 0) {
       usage(EXIT_SUCCESS, stdout);
     } else {
       usage(EXIT_FAILURE, stderr);
     }
 }
}


/*-------------------------------------------------------------------------------*/
/* Print result of the parallel computation and performances                     */
/*-------------------------------------------------------------------------------*/
void PrintResultsAndPerf(double gigaflops, double d1,double d2)
{
 //fprintf(stdout,"- Results:\n");
 fprintf(stdout,"- Results:\n\t C[%d][%d] = %f\n",
         0,SIZE-1,(float) C[0][SIZE-1]);
 fprintf(stdout,"\t C[%d][%d] = %f\n",
         SIZE/2,SIZE/2,(float) C[SIZE/2][SIZE/2]);
 fprintf(stdout,"\t C[%d][%d] = %f\n",
         SIZE-1,0,(float) C[SIZE-1][0]);

 fprintf(stdout,"- Performances:\n");
 fprintf(stdout,"\t Elapsed time of the loop = %f(s)\n",
         (float) d1);
 fprintf(stdout,"\t Gigaflops = %f\n",
         (float) gigaflops);
 fprintf(stdout,"\n\t Total elapsed time of the application = %f(s)\n",
         (float) d2);
 fflush(stdout);

}
