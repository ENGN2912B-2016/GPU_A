#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

#define imin(a,b) (a<b?a:b)

const int nr = 51;
const int nc = 51;
const int N = nr * nc;
const int threadsPerBlock = 256;
const int blocksPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );


__device__ double fun(double x, double y) { 
    const double PI = atan(1.0) * 4;
    return sin(PI*x) * sin(PI*y);
}

//initial condition
__global__ void init(double *d_fout, double *d_unew, double *d_uold, double dx, int nc, int nr) { // initial state filling
    int idx = threadIdx.x + blockIdx.x*blockDim.x; // 2D-grid of pixels, each one being a problem unknown
    int i = idx / nc;
    int j = idx % nc;

    while(idx < (nc) *(nr)){
      double xval = dx * i - 1.0;
      double yval = dx * j - 1.0;
      d_fout[idx] = fun(xval, yval);
      d_unew[idx] = 0.0;
      d_uold[idx] = 0.0; 
      idx += blockDim.x * gridDim.x;

}
     __syncthreads(); 

}
//solve function




__global__ void evolve(double *fout, double *unew, double *uold, double *c, double dx, int nc, int nr) { // initial state filling
    
   __shared__ float cache[threadsPerBlock];
    int idx = threadIdx.x + blockIdx.x*blockDim.x ; // 2D-grid of pixels, each one being a problem unknown
    int cacheIndex = threadIdx.x;
    int left = idx - 1;
    int right = idx + 1;
    int top = idx + nc;
    int bottom = idx - nc;
    double sum = 0;

    while(idx < nc*nr) {
        
        if((idx%nc !=0) && (idx%nc !=nc-1) && (idx > nc)&&(idx < (nc)*(nr-1))){

        unew[idx] = (uold[top] + uold[bottom] + uold[left] + uold[right]) - dx*dx*fout[idx];
        unew[idx] /= (4 + dx*dx);
       sum += pow((unew[idx] - uold[idx]),2);   
        uold[idx] = unew[idx]; 
    
        }   
        idx += blockDim.x * gridDim.x;    
        
    }
    // numerical scheme

   cache[cacheIndex] = sum;
     __syncthreads(); 

     int i = blockDim.x/2;
     while (i != 0) {
       if (cacheIndex < i)
       cache[cacheIndex] += cache[cacheIndex + i];
       __syncthreads();
        i /= 2;
     }

     if (cacheIndex == 0)
        c[blockIdx.x] = cache[0]; 
}  




int main() {

    int Mloop = 1e4;
    double error = 1e-10;
    double  *unew, *fout, *d_uold, *d_unew, *d_fout, *dev_partial_c, *partial_c;
    unew = (double*)malloc(nr*nc*sizeof(double));
    fout = (double*)malloc(nr*nc*sizeof(double));
    partial_c = (double*)malloc(blocksPerGrid*sizeof(double));
    cudaMalloc(&d_uold, nr*nc*sizeof(double)); 
    cudaMalloc(&d_unew, nr*nc*sizeof(double)); 
    cudaMalloc(&d_fout, nr*nc*sizeof(double)); 
    cudaMalloc(&dev_partial_c, blocksPerGrid*sizeof(double)) ;
    

    double dx = 2.0/static_cast<double>(nc - 1);
    init<<<blocksPerGrid, threadsPerBlock>>>(d_fout, d_unew, d_uold, dx, nc, nr);//initialize 
    int count = 0;
    
    double c = 1;
    while(count < Mloop && sqrt(c) > error){
        count += 1;
        evolve<<<blocksPerGrid, threadsPerBlock>>>(d_fout, d_unew, d_uold, dev_partial_c, dx, nc, nr);
        if (count % 100 == 99){
        cudaMemcpy( partial_c, dev_partial_c,blocksPerGrid*sizeof(double),cudaMemcpyDeviceToHost ) ;
         c = 0;
         for (int i=0; i<blocksPerGrid; i++) {
            c += partial_c[i];
        //    std::cout << "error: " << partial_c[i] << std::endl;
         }
         c = c / (nc*nr);
       }
         
         
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
    cudaFree(dev_partial_c);
    free(unew);
    free(partial_c);
    return 0;
    
}
