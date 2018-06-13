CC=mpicc
CFLAGS= -O3 -Wall -fopenmp -funroll-loops 
LDFLAGS= 
SOURCES= Init.c Calcul.c main.c 
EXECNAME= MatrixProduct

all:
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXECNAME) $(SOURCES) -lm -lopenblas

clean:
	rm -f *.o core $(EXECNAME)



