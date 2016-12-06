// GPU acceleration based on OpenACC directives
// to solve Helmholtz problem has been implemented,
// parallel loop region,data region,collapse loops
// and change the number of workers and gangs are
// implemented in this code.
//
// By Shihong Li, Dec 2016

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <algorithm>
#include "timer.h"

#define M_PI  3.14159265358979323846  /* pi */

// RHS functions to calculate the RHS values of the equation
double RHS(double x, double y) {
    double val;
    val = sin(M_PI*x) * sin(M_PI*y);
    return val;
}

int main(int argc, char** argv) {
    
    int nr, nc; // # of rows and # of columns
    
    // Initialize total number of grid points
    if (argc > 1){
        nr = atoi(argv[1]);
        nc = atoi(argv[2]);
    }
    else{
        nr = 201;
        nc = 201;
    }
    
// Data size
    float uold[nr][nc];
    float unew[nr][nc];
    float fout[nr][nc];
    
// Allocate memory
    memset(uold, 0, nr*nc*sizeof(float));
    memset(unew, 0, nr*nc*sizeof(float));
    memset(fout, 0, nr*mc*sizeof(float));
// Initialization
    double N = static_cast<double>(nr*nc);// total number of grid points
    double dx = 2.0/static_cast<double>(nc - 1);
    double xval, yval;
    
    for (int i = 0; i < nc; i++){
        for (int j = 0; j < nr; j++) {
            xval = dx * i - 1.0;
            yval = dx * j - 1.0;
            fout[i][j] = RHS(xval, yval);
        }
    }
    
    printf("Helmholtz calculation: %d x %d mesh \n", nr,nc);
    
    StartTimer();

    const int iter_max = 1e4 ;
    int iter = 0;
    const double tol= 1e-3;
    double err= 1.0;
    
// Obtain the solution in current iteration
    
    #pragma acc data copy(uold,fout), create(unew)
    while ( err > tol && iter < iter_max){
        
      #pragma acc kernels
      {
        err = 0.0;
        #pragma omp parallel for shared(nr,nc,unew,uold,fout)
        for (int i = 1; i < nc - 1; i++) {
            #pragma acc loop gang(8) vector(32)
            for (int j = 1; j < nr - 1; j++) {
                unew[i][j] = (uold[i+1][j] + uold[i-1][j] + uold[i][j-1] + uold[i][j+1]) - dx*dx*fout[i][j];
                unew[i][j] /= (4.0 + dx*dx);
                err += pow((unew[i][j] - uold[i][j]),2);
             }
         }
    
// Swap the new solution to the old solution sets for next iteration
       #pragma omp parallel for shared(nr,nc,unew,uold)
        for (int i = 1; i < nc - 1; i++) {
            #pragma acc loop gang(8) vector(32)
            for (int j = 1; j < nr - 1; j++) {
               
               uold[i][j] = unew[i][j];
            }
         }
       }
    
    if(iter % 100 = 0 ) printf("%5d, %0.6fn", iter,err);
    
    iter++;
    
    }
    
    double runtime = GetTImer();
    
    printf(" total: %f sn", runtime / 1000 );
    return 0;
}
