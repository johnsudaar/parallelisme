/*********************************************************************************/
/* Matrix product program with MPI on a virtual ring of processors               */
/* S. Vialle - October 2014                                                      */
/*********************************************************************************/

#ifndef __MAIN__
#define __MAIN__


/*-------------------------------------------------------------------------------*/
/* CONSTANTS.                                                                    */
/*-------------------------------------------------------------------------------*/

#define DEFAULT_NB_THREADS  1                    /* Default: run 1 thread        */
#define DEFAULT_KERNEL_ID   0                    /* Default: run user kernel     */
#define DEFAULT_NB_LIGCOL   1024                 /* Default: 1024 lines and cols */


/*-------------------------------------------------------------------------------*/
/* Global variable declarations.                                                 */
/*-------------------------------------------------------------------------------*/

/* Matrixes: C = A.B                                                             */
/* We use the Transposed B matrix, in place of B, to improve cache memory usage. */
extern double *A_Slice;                        /* Slices of matrixes (C = AxB) */
extern double *A_Slice_buf;
extern double *B_Slice;
extern double *TB_Slice;
extern double *C_Slice;

extern unsigned long SIZE;                    /* Matrix side                   */
extern unsigned long LOCAL_SIZE;              /* Nb of lines on 1 process      */

/* Global variables for the management of the parallelisation.                   */
/* Need to be initialized dynamically: set to dummy values for the moment.       */
extern int Me;                                  /* Processor coordinates.       */
extern int NbPE;                                /* Number of processors         */

/* Global variables for the management of the result and performance printing.   */
extern unsigned long PrinterPE;      /* Processor hosting the central elt of C.*/
extern unsigned long PrintedElt_i;   /* Coordinates of the central elt of C in */ 
extern unsigned long PrintedElt_j;   /* its host processor.                               */

/* Global variables to control OpenMP and kernel computations.                   */
extern int NbThreads;                           /* Nb of OpenMP threads         */
extern int KernelId;                            /* Kernel Id                    */


/*-------------------------------------------------------------------------------*/
/* Global functions.                                                             */
/*-------------------------------------------------------------------------------*/
void ComputationAndCirculation();
void OneStepCirculation(int step);
int main(int argc, char *argv[]);


#endif

// END
