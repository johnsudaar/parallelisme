/*********************************************************************************/
/* Matrix product program with MPI on a virtual ring of processors               */
/* S. Vialle - January 2017                                                      */
/*********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <openblas/cblas.h>

#include "main.h"
#include "Calcul.h"


/*-------------------------------------------------------------------------------*/
/* Sequential product of local matrixes (optimized seq product).                 */
/*-------------------------------------------------------------------------------*/
void OneSeqLocalProduct(int step, int firstLinAs_th, int nbLinAs_th, double *As)
{
  int OffsetStepLinC;
  int i, j, k;

  // Compute the current step offset, in the MPI program, to access right C lines
  OffsetStepLinC = ((step + Me) % NbPE) * LOCAL_SIZE;; // TO DO 

  switch (KernelId) {

  // Kernel 0 : Optimized code implemented by an application developer
  case 0 :
    for (i = firstLinAs_th; i < firstLinAs_th + nbLinAs_th; i++) {
      int iAoff = i*SIZE;
      for (j = 0; j < LOCAL_SIZE; j++) {
        int jTBoff = j*SIZE;
        double accu[8];
        accu[0] = accu[1] = accu[2] = accu[3] =  accu[4] =  accu[5] =  accu[6] =  accu[7] = 0.0;
        for (k = 0; k < (SIZE/8)*8; k += 8) {
           accu[0] += As[iAoff+k+0] * TB_Slice[jTBoff+k+0];
           accu[1] += As[iAoff+k+1] * TB_Slice[jTBoff+k+1];
           accu[2] += As[iAoff+k+2] * TB_Slice[jTBoff+k+2];
           accu[3] += As[iAoff+k+3] * TB_Slice[jTBoff+k+3];
           accu[4] += As[iAoff+k+4] * TB_Slice[jTBoff+k+4];
           accu[5] += As[iAoff+k+5] * TB_Slice[jTBoff+k+5];
           accu[6] += As[iAoff+k+6] * TB_Slice[jTBoff+k+6];
           accu[7] += As[iAoff+k+7] * TB_Slice[jTBoff+k+7];
        }
        for (k = (SIZE/8)*8; k < SIZE; k++) {
           accu[0] += As[iAoff+k] * TB_Slice[jTBoff+k];
        }
        C_Slice[(i+OffsetStepLinC)*LOCAL_SIZE+j] = accu[0] + accu[1] + accu[2] + accu[3] +
                                                   accu[4] + accu[5] + accu[6] + accu[7];
      }
    }
    break;

  // Kernel 1 : Very optimized computing kernel implemented in a HPC library
  case 1 :
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nbLinAs_th, LOCAL_SIZE, SIZE,
                1.0, &As[firstLinAs_th*SIZE], SIZE,
                B_Slice, LOCAL_SIZE,
                0.0, &C_Slice[(OffsetStepLinC+firstLinAs_th)*LOCAL_SIZE], LOCAL_SIZE);
    break;

  default :
    fprintf(stderr,"Error: kernel %d not implemented!\n",KernelId);
    exit(EXIT_FAILURE);
    break;
  }
}
