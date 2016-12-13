// GPU acceleration based on OpenACC directives
// to solve Helmholtz problem has been implemented.
//
// This is the final version of the code, below features
// are presented:
//
// 1. Parallel loop in OpenACC
// 2. Collapse loops to use longer vectors
// 3. Manage data movement
// 4. Change number of gangs and workers
//
// By Shihong Li, Dec 2016

#include <string.h>
#include <math.h>
#include "timer.h"
#include <iostream>
#include <stdio.h>

using namespace std;

#define M_PI  3.14159265358979323846  /* pi */
#define NM 201
#define NN 201

// Data size
float uold[NM][NN];
float unew[NM][NN];
float fout[NM][NN];

int main(int argc, char** argv) {
    
    const int nr = NM;
    const int nc = NN;
    const int iter_max = 1e5 ;
    const double tol = 1e-5;
    double err =  1.0;
    // Allocate memory
    
    memset(uold, 0, nr*nc*sizeof(float));
    memset(unew, 0, nr*nc*sizeof(float));
    memset(fout, 0, nr*nc*sizeof(float));
    // Initialization
    printf("Helmholtz calculation: %d x %d mesh \n", nr,nc);
    
    StartTimer();
 
        float dx = 2.0/static_cast<float>(nc - 1);
        float xval, yval;
    
        for (int i = 0; i < nc; i++){
            for (int j = 0; j < nr; j++) {
                xval = dx * i - 1.0;
                yval = dx * j - 1.0;
                fout[i][j] = sin(M_PI*xval) * sin(M_PI*yval);
            }
        }
    
    
    int iter = 0;
    // Obtain the solution in current iteration
    
    #pragma acc data copy(uold,fout) create(unew)
    while ( sqrt(err) > tol && iter < iter_max){
        err = 0.0;
        #pragma acc parallel loop collapse(2) reduction(+:err)\
         gang worker num_workers(4) vector_length(32)
             #pragma omp parallel for shared(nr, nc, unew, uold,fout)
            for (int i = 1; i < nc - 1; i++) {
                for (int j = 1; j < nr - 1; j++) {
                    unew[i][j] = (uold[i+1][j] + uold[i-1][j] + uold[i][j-1] + uold[i][j+1]- dx*dx*fout[i][j]) /(4.0 + dx*dx) ;
                    err += pow(unew[i][j] - uold[i][j],2);
                }
            }
        // Swap the new solution to the old solution sets for next iteration
        #pragma acc parallel loop collapse(2) \
        gang worker num_workers(4) vector_length(32)
             #pragma omp parallel for shared(nr, nc, unew, uold,fout)
            for (int i = 1; i < nc - 1; i++) {
                for (int j = 1; j < nr - 1; j++) {
                    
                    uold[i][j] = unew[i][j];
                }
            }
        
        if(iter % 100 == 0 ) printf("%5d, %0.10e\n", iter,sqrt(err));
        
        iter++;
        
    }
    double runtime = GetTimer();
    
    printf(" total runtime: %f \n", runtime / 1000 );
    
}
