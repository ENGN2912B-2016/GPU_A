#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <algorithm> 

#define imin(a,b) (a<b?a:b)
const int threadsPerBlock = 256;
texture<float> tex_fout;
texture<float> tex_uold;
texture<float> tex_unew;


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
//solve function


__global__ void error1(float *c, int nc, int nr) {
  __shared__ float cache[threadsPerBlock];
  int idx = threadIdx.x + blockIdx.x*blockDim.x ; // 2D-grid of pixels, each one being a problem unknown
  int cacheIndex = threadIdx.x;
   float sum = 0;
    while(idx < nr*nc) {
         
        sum += pow((tex1Dfetch(tex_uold,idx) - tex1Dfetch(tex_unew,idx)),2);    
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

__global__ void evolve(float *unew, bool flag, float dx, int nc, int nr) { // initial state filling
      
    int idx = threadIdx.x + blockIdx.x*blockDim.x ; // 1D-grid of pixels, each one being a problem unknown
    int left, right, top, bottom;
    float t, l, cur, r, b;
    // numerical scheme
    while(idx < nr*nc) {
        
        
        if((idx%nc !=0) && (idx%nc !=nc-1) && (idx > nc)&&(idx < (nc)*(nr-1))){
        
        left = idx - 1;
        right = idx + 1;
        top = idx - nc;
        bottom = idx + nc;
        if (flag) {
          t = tex1Dfetch(tex_uold,top);
          l = tex1Dfetch(tex_uold,left);
          r = tex1Dfetch(tex_uold,right);
          b = tex1Dfetch(tex_uold,bottom);
          cur = tex1Dfetch(tex_fout,idx);
        }
        else{
          t = tex1Dfetch(tex_unew,top);
          l = tex1Dfetch(tex_unew,left);
          r = tex1Dfetch(tex_unew,right);
          b = tex1Dfetch(tex_unew,bottom);
          cur = tex1Dfetch(tex_fout,idx);

        }
        
        unew[idx] = (t + b + l + r) - dx*dx*cur;
        unew[idx] /= (4 + dx*dx);
    
        }   
        idx += blockDim.x * gridDim.x;  
        
    }
}  



int main(int argc, char** argv) {
  
  int nr, nc;
      
    if(argc>1){
     nr = atoi( argv[1] );
     nc = atoi( argv[2] );
   }
    else{
     nr = 201;
     nc = 201;
    }

    float N = float(nr)*float(nc);
    int blocksPerGrid = imin( 32, (N+threadsPerBlock-1) / threadsPerBlock );
  
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    int Mloop = 1e5;
    float error = 1e-10;
    float  *unew, *d_uold, *d_unew, *d_fout, *dev_partial_c, *partial_c;
    unew = (float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(blocksPerGrid*sizeof(float));
    cudaMalloc(&d_uold, N*sizeof(float)); 
    cudaMalloc(&d_unew, N*sizeof(float)); 
    cudaMalloc(&d_fout, N*sizeof(float)); 
    cudaMalloc(&dev_partial_c, blocksPerGrid*sizeof(float)) ;

    cudaBindTexture( NULL, tex_fout, d_fout, N*sizeof(float) );
    cudaBindTexture( NULL, tex_uold, d_uold, N*sizeof(float) );
    cudaBindTexture( NULL, tex_unew, d_unew, N*sizeof(float) );
    
    
    float dx = 2.0/static_cast<float>(nc - 1);
   
    init<<<blocksPerGrid, threadsPerBlock>>>(d_fout, d_unew, d_uold, dx, nc, nr);//initialize 
    int count = 0;
    
    float c = 1;
    bool flag = 1;
    while(count < Mloop && sqrt(c) >= error){
        count += 1;
        if (flag)
          evolve<<<blocksPerGrid, threadsPerBlock>>>(d_unew, flag, dx, nc, nr);
        else
          evolve<<<blocksPerGrid, threadsPerBlock>>>(d_uold, flag, dx, nc, nr);
        flag = !flag;
        if (count % 500 == 499){
        error1<<<blocksPerGrid, threadsPerBlock>>>( dev_partial_c, nc, nr);
        cudaMemcpy( partial_c, dev_partial_c, blocksPerGrid*sizeof(float), cudaMemcpyDeviceToHost ) ;
        c = 0;
        for (int i=0; i<blocksPerGrid; i++) {
            c += partial_c[i];
      //      std::cout << "error: " << partial_c[i] << std::endl;
         }
         c = c / N;
       }
              
    }
    

    //copy the memeory back 
    
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "runtime: %3.1f ms\n", elapsedTime );

    cudaEventRecord( start, 0 );
    cudaMemcpy(unew, d_unew, N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime, start, stop );
    printf( "copy unew back to cpu runtime: %3.1f ms\n", elapsedTime );
    

    std::cout << "iteration: " << count << std::endl;
    std::ofstream out;
    out.open("uend_v2.txt");
    
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < nr; j++) {
            out << unew[i*nc + j] << ' ';
        }
        out << '\n';
    }


    
    cudaFree( d_uold );
    cudaFree( d_unew );
    cudaFree( d_fout );
    cudaFree( dev_partial_c );
    cudaUnbindTexture( tex_uold );
    cudaUnbindTexture( tex_unew );
    cudaUnbindTexture( tex_fout );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    free( unew );
    free( partial_c );
    return 0;
}
