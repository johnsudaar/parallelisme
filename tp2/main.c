/*********************************************************************************/
/* Matrix product program with MPI on a virtual ring of processors               */
/* S. Vialle - October 2014                                                      */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#include "main.h"
#include "Init.h"
#include "Calcul.h"


/*-------------------------------------------------------------------------------*/
/* Global variable declarations.                                                 */
/*-------------------------------------------------------------------------------*/

/* Matrixes: C = A.B                                                             */
/* We use the Transposed B matrix, in place of B, to improve cache memory usage. */
double *A_Slice;                                /* A Matrix.                    */
double *A_Slice_buf;                            /* A Matrix buffer              */
double *B_Slice;                                /* B Matrix.                    */
double *TB_Slice;                               /* Transposed B Matrix.         */
double *C_Slice;                                /* C matrix (result matrix).    */

unsigned long SIZE;                            /* Matrix side                   */
unsigned long LOCAL_SIZE;                      /* Nb of lines on 1 process      */

/* Global variables for the management of the parallelisation.                   */
/* Need to be initialized dynamically: set to dummy values for the moment.       */
int Me = -1;                                    /* Processor rank                */
int NbPE = -1;                                  /* Number of processors          */

/* Global variables for the management of the result and performance printing.   */
unsigned long PrinterPE = 0;          /* Processor hosting the central elt of C.*/
unsigned long PrintedElt_i = 0;       /* Coordinates of the central elt of C in */
unsigned long PrintedElt_j = 0;       /* its host processor.                    */

/* Global variables to control computations.                                     */
int NbThreads = -1;
int KernelId = -1;


/*-------------------------------------------------------------------------------*/
/* Parallel computation: local computations and data circulations.               */
/*-------------------------------------------------------------------------------*/
void ComputationAndCirculation()
{
  int step = 0;
  omp_set_num_threads(NbThreads);
  #pragma omp parallel private(step)
  {
    for (step = 0; step < NbPE; step++) {

      // Parallel computation
      int thId = omp_get_thread_num();
      int thNb = omp_get_num_threads();

      #pragma omp single
      {
        memcpy(A_Slice_buf, A_Slice, sizeof(double)*SIZE*LOCAL_SIZE);
      }

      if(thId == thNb - 1) {
        OneStepCirculation(step);
      } else {
        int quotient = LOCAL_SIZE/(thNb - 1);
        int reste = LOCAL_SIZE % (thNb - 1);

        int firstLinAs_th = 0;
        int nbLinAs_th = 0;

        if (thId < reste) {
          nbLinAs_th = quotient + 1;
          firstLinAs_th = (quotient+1)*thId;
        } else {
          nbLinAs_th = quotient;
          firstLinAs_th = quotient*thId + reste;
        }

        //printf("%d/%d: firstLinAs_th = %d, nbLinAs_th = %d\n",Me, NbPE, firstLinAs_th, nbLinAs_th);
        OneSeqLocalProduct(step,firstLinAs_th,nbLinAs_th, A_Slice_buf);
      }
      // MPI communications
      #pragma omp barrier
    }
  }
}


/*-------------------------------------------------------------------------------*/
/* Elementary circulation of A and B.                                            */
/*-------------------------------------------------------------------------------*/
void OneStepCirculation(int step)
{
  MPI_Status status;

  unsigned long src = (Me + 1) % NbPE;// Next
  unsigned long dst = (Me - 1 + NbPE) % NbPE;// Prec
  /*MPI_Sendrecv_replace(A_Slice, SIZE * LOCAL_SIZE, MPI_DOUBLE,
    dst, step,
    src, step,
    MPI_COMM_WORLD,&status);*/
  MPI_Sendrecv(A_Slice_buf, SIZE * LOCAL_SIZE, MPI_DOUBLE, dst, step,
               A_Slice, SIZE * LOCAL_SIZE, MPI_DOUBLE, src, step,
               MPI_COMM_WORLD, &status);
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
    MPI_Init(&argc,&argv);                        /* MPI initialisation.         */
    td2 = MPI_Wtime();                            /* Start app. time measurement.*/
    ProcessorInit();             /* TO COMPLETE *//* Important init on the proc. */
    CommandLineParsing(argc,argv);                /* Cmd line parsing.           */
    ProcessorPostInit();                          /* Postinit.                   */
    LocalMatrixAllocInit();                       /* Initialization of the data  */
    omp_set_num_threads(NbThreads);               /* Max nb of threads/node.     */

    /* Matrix product computation ---------------------------------------------- */
    if (Me == PrinterPE) {
      fprintf(stdout,"Product of two square matrixes of %lux%lu doubles:\n",
      SIZE,SIZE);
      fprintf(stdout,"- Number of MPI processes: %d\n",NbPE);
      fprintf(stdout,"- Max number of OpenMP threads per process: %d\n", NbThreads);
      fprintf(stdout,"- Kernel Id: %d\n",KernelId);
      fprintf(stdout,"- Parallel computation starts...\n");
      fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);                  /* Start comp. time measurement*/
    td1 = MPI_Wtime();
    ComputationAndCirculation(); /* TO COMPLETE *//* Parallel Matrix product.    */
    MPI_Barrier(MPI_COMM_WORLD);                  /* End of all. time measures.  */
    tf1 = MPI_Wtime();                            /* - end of comp. time measure.*/
    tf2 = MPI_Wtime();                            /* - end of app. time measure. */

    /* Performance computation, results and performance printing --------------- */
    d1 = tf1 - td1;                               /* Elapsed comp. time.         */
    d2 = tf2 - td2;                               /* Elapsed app. time.          */
    gigaflops = (2.0*pow(SIZE,3))/d1*1E-9;        /* Performance achieved.       */
    PrintResultsAndPerf(gigaflops,d1,d2);         /* Results and perf printing   */

    /* End of the parallel program --------------------------------------------- */
    Cleanup();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();                               /* End of the MPI usage.       */
    return(EXIT_SUCCESS);
  }
