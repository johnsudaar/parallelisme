/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2016                                                     */
/*********************************************************************************/

#ifndef __MATPROD_INIT__
#define __MATPROD_INIT__


void LocalMatrixInit(void);                      // Data init

void usage(int ExitCode, FILE *std);             // Cmd line parsing and usage
void CommandLineParsing(int argc, char *argv[]); 

void PrintResultsAndPerf(double gigaflops, double d1,double d2); // Res printing


#endif

// END
