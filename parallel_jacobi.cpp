#include <mpi.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

std::pair<int,int> GetRankID(int r, int col) {
    std::pair<int,int> rankID;
    rankID.first = r / col;
    rankID.second = r % col;
    return rankID;
}

std::vector<int> GetMMRC(int N, int nrow, int ncol) {
    std::vector<int> mmrc(4,0);
    int minrowN = N / nrow;
    int mincolN = N / ncol;
    int maxrowN = N - (nrow - 1) * minrowN;
    int maxcolN = N - (ncol - 1) * mincolN;
    mmrc[0] = minrowN;
    mmrc[1] = mincolN;
    mmrc[2] = maxrowN;
    mmrc[3] = maxcolN;
    return mmrc;
}

std::pair<int,int> GetLocalRC(int N, int nrow, int ncol, std::pair<int,int> rankID) {
    std::pair<int,int> localRC;
    std::vector<int> mmrc = GetMMRC(N,nrow,ncol);
    int minrowN = mmrc[0];
    int mincolN = mmrc[1];
    int maxrowN = mmrc[2];
    int maxcolN = mmrc[3];
    if (rankID.first < nrow - 1)
        localRC.first = minrowN;
    else
        localRC.first = maxrowN;
    if (rankID.second < ncol - 1)
        localRC.second = mincolN;
    else
        localRC.second = maxcolN;
    return localRC;
}

std::vector<int> GetNeigh(int nrow, int ncol, int rank) {
    std::vector<int> neighVec(4,-1);
    neighVec[0] = rank - 1;
    neighVec[2] = rank + 1;
    neighVec[1] = rank + ncol;
    neighVec[3] = rank - ncol;
    std::pair<int,int> p = GetRankID(rank, ncol);
    if (p.first == 0) neighVec[3] = rank + ncol * (nrow - 1);
    if (p.first == nrow - 1) neighVec[1] = rank - ncol * (nrow - 1);
    if (p.second == 0) neighVec[0] = rank + ncol - 1;
    if (p.second == ncol - 1) neighVec[2] = rank - (ncol - 1);
    return neighVec;
}

double fun(double x, double y) {
    double val;
    val = sin(M_PI * x) * sin(M_PI * y);
    return val;
}

double CalNewU(double u1, double u2, double u3, double u4,double f,double dx) {
    double unew;
    unew = (u1+ u2+ u3+ u4) - dx*dx*f;
    unew /= (4 + dx*dx);
    return unew;
}

inline int index(int i, int j, int N) {
    return i*N+j;
}

int main(){
    MPI_Init ( NULL, NULL );

    std:: ofstream out;
    out.open("out.txt");

    int nrow = 1, ncol = 4;
    int N = 12;
    double hL = 1, L = 2;
    double *unew = new double[N*N];
    double *u = new double[N*N];
    double *fout = new double[N*N];
    double dx = L / static_cast<double>(N - 1);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fout[index(i,j,N)] = 0.0;
            unew[index(i,j,N)] = 0.0;
            u[index(i,j,N)] = 0.0;
        }
    }

    int rank, nproc;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &nproc );

    // get the row and col of a rank
    std::pair<int,int> rankID = GetRankID(rank, ncol);
    int rowID = rankID.first;
    int colID = rankID.second;

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Status status;
    MPI_Request request;

    // get #s of nodes in 2 directions
    std::pair<int,int> localRC = GetLocalRC(N, nrow, ncol, rankID);
    int localRowN = localRC.first;
    int localColN = localRC.second;

    std::vector<int> nei = GetNeigh(nrow,ncol,rank);
/*    std::cout << "This is rank: " << rank << ". My neighbour are ";
    for (int i = 0; i < 4; i++)
        std::cout << nei[i] << ' ';
    std::cout << std::endl;*/
    // cout infomation
/*    std::cout << "This is rank: " << rank << ", row ID: " << rowID << ", col ID: " << colID
        << ", local row number: " << localRowN << ", local col number: " << localColN << std::endl;*/

    // store the info from your neighbourings
    double *uleft = new double[localRowN], *uright = new double[localRowN];
    double *uup = new double[localColN], *udown = new double[localColN];
    for (int i = 0; i < localRowN; i++) {
        uleft[i] = 0; uright[i] = 0;
    }
    for (int i = 0; i < localColN; i++) {
        uup[i] = 0; udown[i] = 0;
    }

    double tbegin, tend;
    tbegin = MPI_Wtime();

    std::vector<int> mmrc = GetMMRC(N,nrow,ncol);
    int minrowN = mmrc[0], mincolN = mmrc[1];
    int maxrowN = mmrc[2], maxcolN = mmrc[3];
    // calculate f for each node
    for (int i = 0; i < localRowN; i++) {
        for (int j = 0; j < localColN; j++) {
            double xval = dx * static_cast<double>(rowID*minrowN + i) - 1.0;
            double yval = dx * static_cast<double>(colID*mincolN + j) - 1.0;
            fout[index(i,j,N)] = fun(xval, yval);
        }
    }

    int Mloop = 1e4, count = 0;
    double tol = 1e-4, errsum = 1.0, localerr;

    while (count < Mloop && errsum > tol) {
        count++;
        for (int i = 1; i < localRowN - 1; i++) {
            for (int j = 1; j < localColN - 1; j++) {
                unew[index(i,j,N)] = CalNewU(u[index(i+1,j,N)], u[index(i-1,j,N)],u[index(i,j-1,N)], u[index(i,j+1,N)],
                        fout[index(i,j,N)],dx);
            }
        }
        unew[index(0,0,N)] = CalNewU(uleft[0],udown[0], u[index(0,1,N)], u[index(1,0,N)],
                fout[index(0,0,N)], dx);

        unew[index(0,localColN-1,N)] = CalNewU(uright[0], udown[localColN-1], u[index(0,localColN-2,N)], u[index(1,localColN-1,N)],
                fout[index(0,localColN-1,N)], dx);

        unew[index(localRowN-1,0,N)] = CalNewU(uleft[localRowN-1], uup[0], u[index(localRowN-1,1,N)], u[index(localRowN-2,0,N)],
                fout[index(localRowN-1,0,N)],dx);

        unew[index(localRowN-1,localColN-1,N)] = CalNewU(uright[localRowN-1], uup[localColN-1], u[index(localRowN-1,localColN-2,N)], u[index(localRowN-2,localColN-1,N)],
                fout[index(localRowN-1,localColN-1,N)], dx);

        for (int i = 1; i < localRowN - 1; i++) {
            unew[index(i,0,N)] = CalNewU(uleft[i],u[index(i,1,N)],u[index(i+1,0,N)],u[index(i-1,0,N)],
                    fout[index(i,0,N)],dx);
            unew[index(i,localColN-1,N)] = CalNewU(uright[i],u[index(i,localColN-2,N)],u[index(i+1,localColN-1,N)],u[index(i-1,localColN-1,N)],
                    fout[index(i,localColN-1,N)],dx);
        }

        for (int j = 1; j < localColN - 1; j++) {
            unew[index(0,j,N)] = CalNewU(udown[j],u[index(0,j+1,N)],u[index(0,j-1,N)],u[index(1,j,N)],
                    fout[index(0,j,N)],dx);
            unew[index(localRowN-1,j,N)] = CalNewU(uup[j],u[index(localRowN-2,j,N)],u[index(localRowN-1,j-1,N)],u[index(localRowN-1,j+1,N)],
                    fout[index(localRowN-1,j,N)],dx);
        }

        // ensure 0 boundary condition
        for (int i = 0; i < localRowN; i++) {
            if (colID == 0){
                unew[index(i,0,N)] = 0;
            }
            else if(colID == ncol - 1) {
                unew[index(i,localColN-1,N)] = 0;
            }
        }

        for (int j = 0; j  < localColN; j++) {
            if (rowID == 0){
                unew[index(0,j,N)] = 0;
            }
            else if(rowID == nrow - 1) {
                unew[index(localRowN-1,j,N)] = 0;
            }
        }
        // compute error
        localerr = 0.0;
        for (int i = 0; i < localRowN; i++) {
            for (int j = 0; j < localColN; j++) {
               localerr += pow(unew[index(i,j,N)]-u[index(i,j,N)],2);
               u[index(i,j,N)] = unew[index(i,j,N)];
            }
        }

        MPI_Allreduce(&localerr, &errsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        errsum = sqrt(errsum);

        // it seems this step does not work...
        // transfer data by arrays
        double *u0temp = new double[localRowN], *u2temp = new double[localRowN];
        double *u1temp = new double[localColN], *u3temp = new double[localColN];
        for (int i = 0; i < localRowN; i++) {
            u0temp[i] = u[index(i,0,N)];
            u2temp[i] = u[index(i,localColN-1,N)];
        }
        for (int j = 0; j < localColN; j++) {
            u1temp[j] = u[index(localRowN-1,j,N)];
            u3temp[j] = u[index(0,j,N)];
        }
        MPI_Isend(&u0temp[0], localRowN, MPI_DOUBLE, nei[0], 1, MPI_COMM_WORLD, &request);
        MPI_Isend(&u2temp[0], localRowN, MPI_DOUBLE, nei[2], 2, MPI_COMM_WORLD, &request);
        MPI_Recv(&uright[0], localRowN, MPI_DOUBLE, nei[2], 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&uleft[0], localRowN, MPI_DOUBLE, nei[0], 2, MPI_COMM_WORLD, &status);

        MPI_Isend(&u1temp[0], localColN, MPI_DOUBLE, nei[1], 3, MPI_COMM_WORLD, &request);
        MPI_Isend(&u3temp[0], localColN, MPI_DOUBLE, nei[3], 4, MPI_COMM_WORLD, &request);
        MPI_Recv(&uup[0], localColN, MPI_DOUBLE, nei[1], 4, MPI_COMM_WORLD, &status);
        MPI_Recv(&udown[0], localColN, MPI_DOUBLE, nei[3], 3, MPI_COMM_WORLD, &status);

        delete[] u0temp;
        delete[] u2temp;
        delete[] u1temp;
        delete[] u3temp;
    }
    // transfer data from different procs to rank 0 after calculations

    if (rank != 0) {
        for (int i = 0; i < localRowN; i++) {
            for (int j = 0; j < localColN; j++) {
                MPI_Isend(&u[index(i,j,N)], 1, MPI_DOUBLE, 0, 100, MPI_COMM_WORLD, &request);
            }
        }
    }
    else if (rank == 0){
        std::cout << "error = " << errsum << ", count = " << count << std::endl;
        for (int n = 1; n < nproc; n++) {
            std::pair<int,int> rankn = GetRankID(n,ncol);
            std::pair<int,int> rcn = GetLocalRC(N, nrow, ncol, rankn);
            for (int i = 0; i < rcn.first; i++) {
                for (int j = 0; j < rcn.second; j++) {
                    MPI_Recv(&u[index(minrowN*rankn.first + i,mincolN*rankn.second + j,N)], 1,
                            MPI_DOUBLE, n, 100, MPI_COMM_WORLD, &status);
                }
            }
        }
    }
    tend = MPI_Wtime();
    std::cout << "Total wall time: " << tend - tbegin << std::endl;

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                out << u[index(i,j,N)] << ' ';
            }
            out << '\n';
        }
    }
    out.close();
    // find your neighbours
    MPI_Barrier( MPI_COMM_WORLD);
    delete[] udown;
    delete[] uup;
    delete[] uright;
    delete[] uleft;
    delete[] unew;
    delete[] u;
    delete[] fout;
    MPI_Finalize();
    out.close();
    return 0;
}

