#####################################################################################
## CUDA + OpenMP Matrix product lab
## Makefile by S. Vialle, November 2016
#####################################################################################

CPUCC = g++
GPUCC = nvcc

CUDA_TARGET_FLAGS = -arch=sm_20  #-arch=sm_61
CXXFLAGS = -O3  # -DDP
CXXFLAGS += -I/opt/atlas/include -I/opt/cuda/include/  #-I/usr/local/cuda/include
CC_CXXFLAGS = -fopenmp
CUDA_CXXFLAGS = $(CUDA_TARGET_FLAGS)

CC_LDFLAGS =  -fopenmp -L/opt/atlas/lib #-L/usr/lib64/atlas/
CUDA_LDFLAGS = -L/opt/cuda/lib64/  #-L/usr/local/cuda/lib64

CC_LIBS = -lcblas -latlas #-lopenblas
CUDA_LIBS = -lcudart -lcuda -lcublas

CC_SOURCES =  main.cc init.cc
CUDA_SOURCES = gpu-op.cu
CC_OBJECTS = $(CC_SOURCES:%.cc=%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:%.cu=%.o)

EXECNAME = MatrixProduct


all:
	$(CPUCC) -c $(CC_SOURCES) $(CXXFLAGS) $(CC_CXXFLAGS)
	$(GPUCC) -c $(CUDA_SOURCES) $(CXXFLAGS) $(CUDA_CXXFLAGS)
	$(CPUCC) -o $(EXECNAME) $(CC_LDFLAGS) $(CUDA_LDFLAGS) $(CC_OBJECTS) $(CUDA_OBJECTS) $(CUDA_LIBS) $(CC_LIBS)


clean:
	rm -f *.o $(EXECNAME) *.linkinfo *~ *.bak .depend


#Regles automatiques pour les objets
#%.o:  %.cc
#	$(CPUCC)  -c  $(CXXFLAGS) $(CC_CXXFLAGS) $<
#
#%.o:  %.cu
#	$(GPUCC)  -c  $(CXXFLAGS) $(CUDA_CXXFLAGS) $<
