#include "mpi.h"
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <math.h>

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


//=========================================================

#define BLK(x, y) blk[x*blksize+y]
#define RCT(x, y) recvtop[x*blksize+y]
#define RCL(x, y) recvleft[x*blksize+y]

void elim(float* blk, int rank, int blksize) {
    for (int w = 0; w < blksize; w++) {
		for (int i = rank * blksize + w + 1; i < blksize; i++) {
			BLK(w, i) /= BLK(w, rank * blksize + w);
		}
		BLK(w, rank * blksize + w) = 1;
		for (int ww = w + 1; ww < blksize; ww++) {
			for (int i = rank * blksize + w + 1; i < blksize; i++) {
				BLK(ww, i) -= BLK(w, i) * BLK(ww, rank * blksize + w);
			}
			BLK(ww, rank * blksize + w) = 0;
		}
    }
}

int main(int argc,char *argv[]) {
    int rank, size;
    MPI_Status mpis;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int matsize = 512;
    int n_blks = sqrt(size);
    int blksize = matsize / n_blks;
    int rowidx = rank/n_blks;
    int colidx = rank%n_blks;

    float* blk = new float[blksize * blksize];
    float* sendbuf = new float[blksize * blksize];
    float* recvtop = new float[blksize * blksize];
    float* recvleft = new float[blksize * blksize];

    //=========================================================
    // Generate and distribute data among all nodes.
    //=========================================================

    if (rank == 0) {
        float** mat = new_matrix(matsize);
        for (int i=0; i<blksize; i++) {
            for (int j=0; j<blksize; j++) {
                BLK(i, j) = mat[i][j];
            }
        }
        for (int i=0; i<n_blks; i++) {
            for (int j=0; j<n_blks; j++) {
                if (i==0&&j==0) continue;
                for (int ii=0; ii<blksize; ii++) {
                    for (int jj=0; jj<blksize; jj++) {
                        sendbuf[ii*blksize+jj] = mat[ii][jj];
                    }
                }
                MPI_Send(sendbuf, blksize * blksize, MPI_FLOAT, i*n_blks+j, 0, MPI_COMM_WORLD);
            }
        }
        delete_matrix(mat, matsize);
    }
    else {
        MPI_Recv(blk, blksize * blksize, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &mpis);
    }


    //=========================================================
    // Elimination
    //=========================================================


    //=========================================================
    // Node finishes calculation after min(rowidx, colidx)+1 steps
    //=========================================================
    int step = min(rowidx, colidx)+1;
    for (int s=0; s<step; s++) {

        //==================================================
        // Diagonal head
        //==================================================
        if (rowidx == colidx) {
            if (s == rowidx) {

                cout << rank << " head" << endl;

                for (int k=0; k<blksize; ++k) {
                    for (int j=0; j<blksize; j++) {
                        RCL(j, k) = BLK(j, k);
                    }
                    for (int j=k+1; j<blksize; ++j) {
                        BLK(k, j) /= BLK(k, k);
                        RCT(k, j) = BLK(k, j);
                    }
                    BLK(k, k) = 1;
                    for (int i=k+1; i<blksize; i++) {
                        for (int j=k+1; j<blksize; j++)
                            BLK(i, j) -= BLK(i, k) * BLK(k, j);
                       BLK(i, k) = 0;
                    }
                }

                MPI_Send(recvleft, blksize * blksize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
                MPI_Send(recvtop, blksize * blksize, MPI_FLOAT, rank + blk_rows, 0, MPI_COMM_WORLD);
                continue;
            }
        }

        //==================================================
        // Receive data from left or top & forward
        //==================================================
        if (colidx > 0) {
            MPI_Recv(recvleft, blksize * blksize, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &mpis);
            cout << rank << " recvleft" << endl;
            if (colidx < n_blks-1) {
                MPI_Send(recvleft, blksize * blksize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD); // Forward
            }
        }
        if (rowidx > 0) {
            MPI_Recv(recvtop, blksize * blksize, MPI_FLOAT, rank - blk_rows, 0, MPI_COMM_WORLD, &mpis);
            cout << rank << " recvtop" << endl;
            if (rowidx < n_blks-1) {
                MPI_Send(recvtop, blksize * blksize, MPI_FLOAT, rank + blk_rows, 0, MPI_COMM_WORLD); // Forward
            }
        }


        //==================================================
        // If this block is on upper edge (s == rowidx)
        // Perform division
        // Send data to bottom
        //==================================================
        if (s == rowidx) {
            cout << rank << " upper edge" << endl;
            for (int i=0; i<blksize; i++) {
                for (int k=0; k<blksize; k++) {
                    BLK(i, k) /= RCL(k, i);
                    RCT(i, k) = BLK(i, k);
                    for (int j=i+1; j<blksize; j++) {
                        BLK(j, k) /= RCL(k, j);
                        BLK(j, k) -= BLK(i, k);
                    }
                }
            }
            MPI_Send(recvtop, blksize * blksize, MPI_FLOAT, rank + n_blks, 0, MPI_COMM_WORLD);
        }

        //==================================================
        // If this block is on left edge (s == colidx)
        // Perform division & substraction
        // Send data to right
        // Set elements to 0
        //==================================================
        else if (s == colidx) {
            cout << rank << " left edge" << endl;
            for (int i=0; i<blksize; i++) {
                for (int j=0; j<blksize; j++) {
                    RCL(j, i) = BLK(j, i);
                    BLK(j, i) = 0;
                    for (int k=i+1; k<blksize; k++) {
                        BLK(j, k) /= BLK(j, i);
                        BLK(j, k) -= RCT(i, k);
                    }
                }
            }
            MPI_Send(recvleft, blksize * blksize, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        }

        //==================================================
        // Else (internal block)
        //==================================================
        else {
            cout << rank << " internal" << endl;
            for (int i=0;i<blksize;i++) {
                for (int j=0;j<blksize;j++) {
                    for (int k=0; k<blksize; k++) {
                        BLK(j, k) /= RCL(j, i);
                        BLK(j, k) -= RCT(i, k);
                    }
                }
            }
        }

        cout << rank << " done" << endl;

    }


    delete[]blk;
    MPI_Finalize();
    return 0;
}
