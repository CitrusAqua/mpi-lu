#include "mpi.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

using namespace MPI;
using namespace std;


float** new_matrix(int matsize) {
    srand((unsigned)time(0));
    float** mat = new float*[matsize];
    for (int i=0; i<matsize; ++i) {
        mat[i] = new float[matsize];
        for (int j=0; j<matsize; ++j) {
            mat[i][j] = (float)rand() / (RAND_MAX>>8);
        }
    }
    return mat;
}

float** new_zeros(int matsize) {
    float** mat = new float*[matsize];
    for (int i=0; i<matsize; ++i) {
        mat[i] = new float[matsize];
        for (int j=0; j<matsize; ++j) {
            mat[i][j] = 0;
        }
    }
    return mat;
}


void delete_matrix(float** mat, int _size) {
    for (int i=0;i<_size;++i) {
        delete[](mat[i]);
    }
    delete[]mat;
}


//////////////////////////////////////////////////////

#define ROW(x, y) row[x*matsize+y]

int main(int argc,char *argv[]) {
    int rank, size;
    MPI_Status s;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int matsize = 512;
    int width = matsize / size;

    float* row = new float[matsize * width];
    float* sendbuf = new float[matsize * width];
    float* recvbuf = new float[matsize * width];

    // Generate and distribute data among all nodes.
    if (rank == 0) {
        float** mat = new_matrix(matsize);
        for (int w=0; w<width; w++) {
            for (int i=0; i<matsize; i++) {
                ROW(w, i) = mat[w][i];
            }
        }
        for (int i=1; i<size; i++) {
            for (int w=0; w<width; w++) {
                for (int j=0; j<matsize; j++) {
                    sendbuf[w*width+i] = mat[width*i+w][j];
                }
            }
            MPI_Send(sendbuf, matsize * width, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
            cout << i << " Sent!" << endl;
        }
        delete_matrix(mat, matsize);
    }
    else {
        MPI_Recv(row, matsize * width, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &s);
        cout << rank << " Received!" << endl;
    }


    struct timeval tpstart, tpend;
    double timeuse;
    gettimeofday(&tpstart, NULL);


    // Receive data from formar nodes and perform eliminations.
    if (rank != 0) {
        for (int r=0; r<rank; r++) {
            MPI_Recv(recvbuf, matsize * width, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &s);
            if (rank < size-1) {
                MPI_Send(recvbuf, matsize * width, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD); // Forward
            }
            cout << rank << " data from formar nodes received!" << endl;
            for (int w=0; w<width; w++) {
                for (int ww=0; ww<width; ww++) {
                    for (int i=r*width+w+1; i<matsize; i++) {
                        ROW(ww, i) -= recvbuf[w*matsize+i] * ROW(ww, r*width+w);
                    }
                    ROW(ww, r*width+w) = 0;
                }
            }
        }
    }

    cout << rank << " data collect done!" << endl;

    // Perform elimination on own data.
    for (int w = 0; w < width; w++) {
		for (int i = rank * width + w + 1; i < matsize; i++) {
			ROW(w, i) /= ROW(w, rank * width + w);
		}
		ROW(w, rank * width + w) = 1;
		for (int ww = w + 1; ww < width; ww++) {
			for (int i = rank * width + w + 1; i < matsize; i++) {
				ROW(ww, i) -= ROW(w, i) * ROW(ww, rank * width + w);
			}
			ROW(ww, rank * width + w) = 0;
		}
    }

    cout << rank << " elim done!" << endl;

    // Elimination finished. Send data to latter nodes to perform elimination.
    if (rank < size-1) {
        MPI_Send(row, matsize * width, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD);
    }

    // Collect the whole matrix on rank 0.
    if (rank == 0) {
        float** result = new_zeros(matsize);
        for (int w=0; w<width; w++) {
            for (int i=0; i<matsize; i++) {
                result[w][i] = ROW(w, i);
            }
        }
        for (int r=1; r<size; r++) {
            MPI_Recv(recvbuf, matsize * width, MPI_FLOAT, r, 0, MPI_COMM_WORLD, &s);
            for (int w=0; w<width; w++) {
                for (int i=0; i<matsize; i++) {
                    result[r*width+w][i] = recvbuf[w*matsize+i];
                }
            }
        }

        gettimeofday(&tpend,NULL);
        timeuse = 1000000*(tpend.tv_sec-tpstart.tv_sec)+tpend.tv_usec-tpstart.tv_usec;
        cout << "running time: " << timeuse << endl;

        delete_matrix(result, matsize);
    }
    else {
        MPI_Send(row, matsize * width, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }


    delete[]row;
    delete[]sendbuf;
    delete[]recvbuf
    MPI_Finalize();
    return 0;
}
