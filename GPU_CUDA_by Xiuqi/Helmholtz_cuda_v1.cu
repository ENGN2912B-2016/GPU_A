//by Xiuqi Li 2016/12
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <algorithm> 

#define imin(a,b) (a<b?a:b)

//input parameters 
//using 1D threads and blocks 
struct pars {
  int threadsPerBlock;
  int blocksPerGrid;
  int nr;//nr = nc is the number of mesh points on each side of the 2*2 square domain
  int nc;
};


__device__ float fun(float x, float y) {
    const float PI = atan(1.0) * 4; 
    return sin(PI*x) * sin(PI*y);
}

//initial condition
__global__ void init(float *d_fout, float *d_unew, float *d_uold, float dx, int nc, int nr) { // initial state filling
    int idx = threadIdx.x + blockIdx.x*blockDim.x; // 2D-grid of pixels, each one being a problem unknown

    while(idx < (nc) *(nr)){
      int i = idx / nc;
      int j = idx % nc;
      float xval = dx * i - 1.0;
      float yval = dx * j - 1.0;
      d_fout[idx] = fun(xval, yval);
      d_unew[idx] = 0.0;
      d_uold[idx] = 0.0; 
      idx += blockDim.x * gridDim.x;

}

}

//caculate error, error is defined as the L2 norm of (unew - uold)
__global__ void error1(float *unew, float *uold, float *c, int nc, int nr) {
  __shared__ float cache[256];
  int idx = threadIdx.x + blockIdx.x*blockDim.x ; // 1D-grid of pixels, each one being a problem unknown
  int cacheIndex = threadIdx.x;
   float sum = 0;
    while(idx < nr*nc) {
         
        sum += pow((unew[idx] - uold[idx]),2);    
        idx += blockDim.x * gridDim.x;    
        
    }
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

// slove unew given uold
__global__ void evolve(float *fout, float *unew, float *uold, float dx, int nc, int nr) { // initial state filling
      
    int idx = threadIdx.x + blockIdx.x*blockDim.x ; // 1D-grid of pixels, each one being a problem unknown
    int left, right, top, bottom;
   
    // numerical scheme
    while(idx < nr*nc) {
        
        
        if((idx%nc !=0) && (idx%nc !=nc-1) && (idx > nc)&&(idx < (nc)*(nr-1))){
        
        left = idx - 1;
        right = idx + 1;
        top = idx - nc;
        bottom = idx + nc;
        unew[idx] = (uold[top] + uold[bottom] + uold[left] + uold[right]) - dx*dx*fout[idx];
        unew[idx] /= (4 + dx*dx);
    
        }   
        idx += blockDim.x * gridDim.x;  
        
    }
  
}  

//check if input parameters are in correct range, throw error if not 
pars init_pars(int nr, int threads, int blocks){
    pars inputpars;
    if (nr <= 31){
      throw "mesh size should not be less than 31 to get accurate results";
    }    
    else if (nr%2 == 0){
      nr = nr+1;
    }

    if (threads > 256){
      throw "threads per block should not be greater than 256";
    }

    if (blocks > 65535){
      throw "blocks per grid should not be greater than 65535";
    }
   
   inputpars.nr = nr;
   inputpars.nc = nr;
   inputpars.blocksPerGrid = blocks;
   inputpars.threadsPerBlock = threads;
   return inputpars;
}

int main(int argc, char** argv) {
//using 1D threads and blocks 
  
//default parameters
     int nr = 501;
     float N = float(nr)*float(nr);
     int threads = 256;
     int blocks = imin( 10000, (N+threads-1) / threads );
 
     switch (argc) {
           
        case 4:    blocks = atoi( argv[3] ); 
        case 3 :   threads = atoi( argv[2] );
        case 2 :   nr = atoi( argv[1] );
                    break;
        case 1 :    std::cin >> nr >> threads >> blocks;
                    break;
        default :   std::cerr << "Bad number of input parameters!" << std::endl;
                    return(-1);
    }

     if (argc == 2){
      N = float(nr)*float(nr);
      threads = 256;
      blocks = imin( 10000, (N+threads-1) / threads );
     }
     
//error handling 
     pars inputpars;
     try{
      inputpars = init_pars(nr, threads, blocks);
      }catch (const char* msg) {
      std::cerr << msg << std::endl;
      return 0;
   }
     
     nr = inputpars.nr;
     int nc = nr;
     int blocksPerGrid = inputpars.blocksPerGrid;
     int threadsPerBlock = inputpars.threadsPerBlock;
     N = float(nr)*float(nc);

//runtime counter
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    int Mloop = 1e5;//largest iteration times
    float error = 1e-5;//convergence criterion
    //allocate memory
    float  *unew, *d_uold, *d_unew, *d_fout, *dev_partial_c, *partial_c;
    unew = (float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));
    cudaMalloc(&d_uold, N*sizeof(float)); 
    cudaMalloc(&d_unew, N*sizeof(float)); 
    cudaMalloc(&d_fout, N*sizeof(float)); 
    cudaMalloc(&dev_partial_c, blocksPerGrid*sizeof(float)) ;
    
    float dx = 2.0/static_cast<float>(nc - 1);
    //initialization
    init<<<blocksPerGrid, threadsPerBlock>>>(d_fout, d_unew, d_uold, dx, nc, nr);//initialize 
    int count = 0;
    
    float c = 1;
    while(count < Mloop && sqrt(c) >= error){
        count += 1;
        evolve<<<blocksPerGrid, threadsPerBlock>>>(d_fout, d_unew, d_uold, dx, nc, nr);
        std::swap( d_uold, d_unew );
        if (count % 500 == 499){//check error every 500 steps
        error1<<<blocksPerGrid, threadsPerBlock>>>(d_unew, d_uold, dev_partial_c, nc, nr);
        cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost ) ;
        c = 0;
        for (int i=0; i<blocksPerGrid; i++) {
            c += partial_c[i];
      //      std::cout << "error: " << partial_c[i] << std::endl;
         }
       }
              
    }
    
    
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "runtime: %3.1f ms\n", elapsedTime );

    cudaEventRecord( start, 0 );
    cudaMemcpy(unew, d_unew, N*sizeof(float), cudaMemcpyDeviceToHost);//copy results back to cpu
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "copy unew back to cpu runtime: %3.1f ms\n", elapsedTime );
    

    std::cout << "iteration: " << count << std::endl;
    std::ofstream out;
    out.open("uend_v1.txt");
    
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < nr; j++) {
            out << unew[i*nc + j] << ' ';
        }
        out << '\n';
    }


    //free memory
    cudaFree( d_uold );
    cudaFree( d_unew );
    cudaFree( d_fout );
    cudaFree( dev_partial_c );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    free( unew );
    free( partial_c );

    return 0;
    
}
