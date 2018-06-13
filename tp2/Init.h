/*********************************************************************************/
/* Matrix product program with MPI on a virtual ring of processors               */
/* S. Vialle - October 2014                                                      */
/*********************************************************************************/

#ifndef __INIT__
#define __INIT__


void ProcessorInit(void);                        // Processor init
void ProcessorPostInit(void);                    // Processor postinit (after cmd line parsing)
void LocalMatrixAllocInit(void);                 // Data alloc & init
void Cleanup(void);                              // Data cleanup

void usage(int ExitCode, FILE *std);             // Cmd line parsing and usage
void CommandLineParsing(int argc, char *argv[]); 

void PrintResultsAndPerf(double megaflops, double d1,double d2); // Res printing


#endif

// END
