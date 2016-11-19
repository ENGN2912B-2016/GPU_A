#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>


__device__ double fun(double x, double y) { // initial state definition
// This function is only called on the device, and never on the host, hence it has to carry the identifier __device__
    
    return sin(M_PI*x) * sin(M_PI*y);
}
//initial condition
__global__ void init(double *d_fout, double *d_unew, double *d_uold, double dx, int nc, int nr) { // initial state filling
int i = threadIdx.x + blockIdx.x*blockDim.x; // 2D-grid of pixels, each one being a problem unknown
int j = threadIdx.y + blockIdx.y*blockDim.y;
int idx = i + j*blockDim.x*gridDim.x;

if(idx < (nc) *(nr)){
  double xval = dx * i - 1.0;
  double yval = dx * j - 1.0;
  d_fout[idx] = fun(xval, yval);
  d_unew[idx] = 0.0;
  d_uold[idx] = 0.0; 
}

}
//solve function
__global__ void evolve(double *fout, double *unew, double *uold, double dx, int nc, int nr) { // initial state filling
int i = threadIdx.x + blockIdx.x*blockDim.x ; // 2D-grid of pixels, each one being a problem unknown
int j = threadIdx.y + blockIdx.y*blockDim.y ;
int idx = i + j*blockDim.x*gridDim.x;

int left = idx - 1;
int right = idx + 1;
int top = idx + nr;
int bottom = idx - nr;
int Mloop = 1e4;
double error = 1e-3;

    if((idx < (nc)*(nr))&& (i !=0)&&(j !=0)&& (i !=nc-1)&&(j !=nr-1)) {

        unew[idx] = (uold[top] + uold[bottom] + uold[left] + uold[right]) - dx*dx*fout[idx];
        unew[idx] /= (4 + dx*dx);
         uold[idx] = unew[idx];       
        
    }// numerical scheme   
}



int main() {
    //nr is # of row, nc # of column, chose to be the same for convenience.
    int nr = 51;
    int nc = nr;


    double  *unew, *d_uold, *d_unew, *d_fout;
    unew = (double*)malloc(nr*nc*sizeof(double));
    cudaMalloc(&d_uold, nr*nc*sizeof(double)); 
    cudaMalloc(&d_unew, nr*nc*sizeof(double)); 
    cudaMalloc(&d_fout, nr*nc*sizeof(double)); 
    

    double dx = 2.0/static_cast<double>(nc - 1);
    dim3 grid(nr,nc); // grid = 1 x 1 blocks
    dim3 block(1,1); // block = nr x nc threads
    init<<<grid, block>>>(d_fout, d_unew, d_uold, dx, nc, nr);//initialize 
    int count = 0;
    //double sum =0.0;
    
    //iterate 500 times
    while(count < 500){
        evolve<<<grid, block>>>(d_fout, d_unew, d_uold, dx, nc, nr);
        count += 1;
    }

    //copy the memeory back 
    cudaMemcpy(unew, d_unew, nc*nr*sizeof(double), cudaMemcpyDeviceToHost);
    

    std::cout << "iteration: " << count << std::endl;
    std::ofstream out;
    out.open("uend.txt");
    
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < nr; j++) {
            out << unew[i*nc + j] << ' ';
        }
        out << '\n';
    }
    
    cudaFree(d_uold);
    cudaFree(d_unew);
    cudaFree(d_fout);
    return 0;
    
}
