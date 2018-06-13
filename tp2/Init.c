/*********************************************************************************/
/* Matrix product program with MPI on a virtual ring of processors               */
/* S. Vialle - October 2014                                                      */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <openblas/cblas.h>

#include "main.h"
#include "Init.h"


/*-------------------------------------------------------------------------------*/
/* Initialisation of processor coordinates.                                      */
/*-------------------------------------------------------------------------------*/
void ProcessorInit(void)
{
 MPI_Comm_size(MPI_COMM_WORLD,&NbPE);            // MPI Init
 MPI_Comm_rank(MPI_COMM_WORLD,&Me);
 
 openblas_set_num_threads(1);                    // Set OpenBLAS in sequential mode 
}


/*-------------------------------------------------------------------------------*/
/* Initialisation of processor coordinates and matrix slice nb of lines          */
/*-------------------------------------------------------------------------------*/
void ProcessorPostInit(void)
{
/* Compute the nb of lines of A_Slice in each process                            */
 if (SIZE % NbPE != 0) {
   if (Me == 0) {
     fprintf(stderr,"Fatal Error: Nb of lines (%lu) is not multiple of nb of process (%d)\n",
             SIZE, NbPE);
   }
   MPI_Finalize();
   exit(0);
 }
 LOCAL_SIZE = SIZE/NbPE;

/* Intializations of variables to "print the central element of the C matrix".   */
/* Computation of the coordinates of the processor hosting the central element,  */
/* and of the coordinates of the central element (of the global C matrix) inside */
/* its host processor.                                                           */
 PrinterPE = NbPE/2;
 if (NbPE % 2 != 0) {
   PrintedElt_i = SIZE/2;
   PrintedElt_j = LOCAL_SIZE/2;
 } else {
   PrintedElt_i = SIZE/2;
   PrintedElt_j = 0;
 }

/* Check if the configuration of the computations and of the run seem coherent.  */
 if (Me < 0 || NbPE <= 0 || Me >= NbPE) {
   fprintf(stderr,
           "MatrixProd: PE%d Fatal Erorr: Bad intialization of Me and NbPE "
           "variables: Me = %d, NbPE = %d!\n",
           Me, Me, NbPE);
   MPI_Finalize();
   exit(0);
 }
}


/*-------------------------------------------------------------------------------*/
/* Allocation & Initialisation of local matrixes A, B, TB and C                  */
/* Each process initializes its local parts of matrixes: simulates a parallel    */
/* initialization from files on disks.                                           */
/*-------------------------------------------------------------------------------*/
void LocalMatrixAllocInit(void)
{
 unsigned long OffsetA_i, OffsetTB_j;   /* Offset of the local matrix elements */
 unsigned long i, j;                    /* Local matrix indexes                */

/* Matrix allocations                                                            */
 A_Slice = (double *) malloc(sizeof(double)*SIZE*LOCAL_SIZE);
 A_Slice_buf = (double *) malloc(sizeof(double)*SIZE*LOCAL_SIZE);
 B_Slice = (double *) malloc(sizeof(double)*SIZE*LOCAL_SIZE);
 TB_Slice = (double *) malloc(sizeof(double)*SIZE*LOCAL_SIZE);
 C_Slice = (double *) malloc(sizeof(double)*SIZE*LOCAL_SIZE);
 if (A_Slice == NULL || A_Slice_buf == NULL || B_Slice == NULL || 
     TB_Slice == NULL || C_Slice == NULL) {
   fprintf(stderr,"Not enough memory to allocate matrixes on process %d\n", Me);
   MPI_Finalize();
   exit(0);
 }

/* Offset of line and column numbers of the element on the processor.            */
 OffsetA_i = LOCAL_SIZE*Me;
 OffsetTB_j = LOCAL_SIZE*Me;

/* Initialization of the local matrix elements                                   */
 for (i = 0; i < LOCAL_SIZE; i++)
    for (j = 0; j < SIZE; j++)
       A_Slice[i*SIZE+j] = (double) ((i+OffsetA_i)*SIZE + j);

 for (i = 0; i < SIZE; i++)
    for (j = 0; j < LOCAL_SIZE; j++) {
       B_Slice[i*LOCAL_SIZE+j] = (double) (i*SIZE + j + OffsetTB_j);
       TB_Slice[j*SIZE+i] = (double) (i*SIZE + j + OffsetTB_j);
    }

 for (i = 0; i < SIZE; i++)
    for (j = 0; j < LOCAL_SIZE; j++)
       C_Slice[i*LOCAL_SIZE+j] = 0.0;
}


/*-------------------------------------------------------------------------------*/
/* Cleanup datastructures                                                        */
/*-------------------------------------------------------------------------------*/
void Cleanup(void)
{
 free(A_Slice);
 free(B_Slice);
 free(TB_Slice);
 free(C_Slice);
}


/*-------------------------------------------------------------------------------*/
/* Printing pgm usage                                                            */
/*-------------------------------------------------------------------------------*/
void usage(int ExitCode, FILE *std)
{
 if (Me == 0) {
   fprintf(std,"MatrixProduct usage: \n");
   fprintf(std,"\t [-lc <nb of lines and columns> (default %d)]\n",DEFAULT_NB_LIGCOL);
   fprintf(std,"\t [-klc <nb of kilo lines and columns> (default %d)]\n",DEFAULT_NB_LIGCOL/1024);
   fprintf(std,"\t [-k <Kernel Id> (default %d)]\n",DEFAULT_KERNEL_ID);
   fprintf(std,"\t [-h]: print this help\n");
   fprintf(std,"\t [-nt <number of threads> (default %d)]\n",DEFAULT_NB_THREADS);
 }
 MPI_Finalize();
 exit(ExitCode);
}


/*-------------------------------------------------------------------------------*/
/* Command Line parsing.                                                         */
/*-------------------------------------------------------------------------------*/
void CommandLineParsing(int argc, char *argv[])
{
 // Default init
 NbThreads = DEFAULT_NB_THREADS;
 KernelId  = DEFAULT_KERNEL_ID;
 SIZE = DEFAULT_NB_LIGCOL;

 // Check if the configuration of the computations and of the run seem coherent
 if (Me < 0 || NbPE <= 0 || Me >= NbPE) {
   fprintf(stderr,
           "MatrixProd: PE%d Fatal Erorr: Bad intialization of Me and NbPE "
           "variables: Me = %d, NbPE = %d!\n",
           Me, Me, NbPE);
   MPI_Finalize();
   exit(0);
 }

 // Init from the command line
 argc--; argv++;
 while (argc > 0) {
     if (strcmp(argv[0],"-lc") == 0) {
       argc--; argv++;
       if (argc > 0) {
         SIZE = atoi(argv[0]);
         argc--; argv++;
         if (SIZE <= 0) {
           if (Me == 0) 
             fprintf(stderr,"Error: number of lines and columns has to be >= 1!\n");
           MPI_Finalize();
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }
     } else if (strcmp(argv[0],"-klc") == 0) {
       argc--; argv++;
       if (argc > 0) {
         SIZE = 1024*atoi(argv[0]);
         argc--; argv++;
         if (SIZE <= 0) {
           if (Me == 0)
             fprintf(stderr,"Error: number of kilo lines and columns has to be >= 1!\n");
           MPI_Finalize();
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }
     } else if (strcmp(argv[0],"-nt") == 0) {
       argc--; argv++;
       if (argc > 0) {
         NbThreads = atoi(argv[0]);
         argc--; argv++;
         if (NbThreads <= 0) {
           if (Me == 0)
             fprintf(stderr,"Error: number of thread has to be >= 1!\n");
           MPI_Finalize();
           exit(EXIT_FAILURE);
         }
       } else {
         usage(EXIT_FAILURE, stderr);
       }
     } else if (strcmp(argv[0],"-k") == 0) {
       argc--; argv++;
       if (argc > 0) {
         KernelId = atoi(argv[0]);
         argc--; argv++;
         if (KernelId < 0 || KernelId > 1) {
           if (Me == 0)
             fprintf(stderr,"Error: kernel Id has to be an integer in [0;1]\n");
           MPI_Finalize();
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
void PrintResultsAndPerf(double gigaflops, double d1, double d2)
{
 if (Me == NbPE-1) {
   fprintf(stdout,"- Results:\n");
   fprintf(stdout,"\tPE%d: C[%lu][%lu] = %f\n",
           Me,0lu,SIZE-1,
           (float) C_Slice[LOCAL_SIZE-1]);
   fflush(stdout);
 }
 MPI_Barrier(MPI_COMM_WORLD);
 if (Me == PrinterPE) {
   fprintf(stdout,"\tPE%d: C[%lu][%lu] = %f\n",
           Me,PrintedElt_i,LOCAL_SIZE*Me+PrintedElt_j,
           (float) C_Slice[PrintedElt_i*LOCAL_SIZE+PrintedElt_j]);
   fflush(stdout);
 }
 MPI_Barrier(MPI_COMM_WORLD);
 if (Me == 0) {
   fprintf(stdout,"\tPE%d: C[%lu][%lu] = %f\n",
           Me,SIZE-1,0lu,(float) C_Slice[(SIZE-1)*LOCAL_SIZE]);
   fprintf(stdout,"- Performances:\n");
   fprintf(stdout,"\tPE%d: Elapsed time of the loop = %f(s)\n",
           Me,(float) d1);
   fprintf(stdout,"\tPE%d: Total Gigaflops = %f\n",
           Me,(float) gigaflops);
   fprintf(stdout,"\n\tPE%d: Total elapsed time of the application = %f(s)\n",
           Me,(float) d2);
   fflush(stdout);
 }
 MPI_Barrier(MPI_COMM_WORLD);
}
