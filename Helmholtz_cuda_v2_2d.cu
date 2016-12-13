//by Xiuqi 2016/12
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdlib.h>
#include <algorithm> 

#define imin(a,b) (a<b?a:b)
//using texture memory for uold and unew
texture<float> tex_uold;
texture<float> tex_unew;

//input parameters 
//using 2D threads and blocks 
struct pars {
  int threadsPerBlock;//2d block size is threadsPerBlock*threadsPerBlock
  int blocksPerGrid;//2d grid size is blocksPerGrid*blocksPerGrid
  int nr;//nr = nc is the number of mesh points on each side of the 2*2 square domain
  int nc;
};




__device__ float fun(float x, float y) { 
    const float PI = atan(1.0) * 4;
    return sin(PI*x) * sin(PI*y);
}

//initial condition
__global__ void init(float *d_fout, float *d_unew, float *d_uold, float dx, int nc, int nr) { // initial state filling
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * blockDim.x * gridDim.x; // 2D-grid of pixels, each one being a problem unknown

    while(idx < (nc) *(nr)){
      int i = idx / nc;
      int j = idx % nc;
      float xval = dx * i - 1.0;
      float yval = dx * j - 1.0;
      d_fout[idx] = fun(xval, yval);
      d_unew[idx] = 0.0;
      d_uold[idx] = 0.0; 
      idx += blockDim.x*blockDim.y * gridDim.x*gridDim.y;

}

}
//solve function

//caculate error, error is defined as the L2 norm of (unew - uold)
__global__ void error1(float *c, int nc, int nr) {
  __shared__ float cache[256];
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = x + y * blockDim.x * gridDim.x; // 2D-grid of pixels, each one being a problem unknown

  int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x ;
   float sum = 0;
    while(idx < nr*nc) {
         
        sum += pow((tex1Dfetch(tex_uold,idx) - tex1Dfetch(tex_unew,idx)),2);    
        idx += blockDim.x*blockDim.y * gridDim.x*gridDim.y;  
        
    }
     cache[cacheIndex] = sum;
     __syncthreads(); 

     int i = blockDim.x*blockDim.y/2;
     while (i != 0) {
       if (cacheIndex < i)
       cache[cacheIndex] += cache[cacheIndex + i];
       __syncthreads();
        i /= 2;
     }

     if (cacheIndex == 0)
        c[blockIdx.x + blockIdx.y * gridDim.x] = cache[0]; 

}

// slove unew given uold
__global__ void evolve(float *fout, float *unew, bool flag, float dx, int nc, int nr) { // initial state filling
      
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * blockDim.x * gridDim.x; // 2D-grid of pixels, each one being a problem unknown
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
          //cur = tex1Dfetch(tex_fout,idx);
          cur = fout[idx];
        }
        else{
          t = tex1Dfetch(tex_unew,top);
          l = tex1Dfetch(tex_unew,left);
          r = tex1Dfetch(tex_unew,right);
          b = tex1Dfetch(tex_unew,bottom);
          //cur = tex1Dfetch(tex_fout,idx);
          cur = fout[idx];

        }
        
        unew[idx] = (t + b + l + r) - dx*dx*cur;
        unew[idx] /= (4 + dx*dx);
    
        }   
        idx += blockDim.x*blockDim.y * gridDim.x*gridDim.y;
        
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

    if (threads > 16){
      throw "threads dimension per block should not be greater than 16 for 2d block";
    }

    if (blocks > 255){
      throw "blocks dimension per grid should not be greater than 255 for 2d grid";
    }
   
   inputpars.nr = nr;
   inputpars.nc = nr;
   inputpars.blocksPerGrid = blocks;
   inputpars.threadsPerBlock = threads;
   return inputpars;
}



int main(int argc, char** argv) {
  //using 2D threads and blocks 
  
   //default parameters

     int nr = 501;
     float N = float(nr)*float(nr);
     int threads = 16;
     int blocks = imin( 100, (nr+threads-1) / threads );
 
  
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
      threads = 16;
      blocks = imin( 100, (nr+threads-1) / threads );
     }

     pars inputpars;
     try{
      inputpars = init_pars(nr, threads, blocks);
      }catch (const char* msg) {
      std::cerr << msg << std::endl;
      return 0;
   }
     
     nr = inputpars.nr;
     int nc = nr;
     int Grid = inputpars.blocksPerGrid;
     dim3 threadsPerBlock(inputpars.threadsPerBlock, inputpars.threadsPerBlock);
     N = float(nr)*float(nc);
     dim3 blocksPerGrid(Grid, Grid);
  
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    cudaEventRecord( start, 0 );

    int Mloop = 1e5;
    float error = 1e-5;
    float  *unew, *d_uold, *d_unew, *d_fout, *dev_partial_c, *partial_c;
    unew = (float*)malloc(N*sizeof(float));
    partial_c = (float*)malloc(Grid*Grid*sizeof(float));
    cudaMalloc(&d_uold, N*sizeof(float)); 
    cudaMalloc(&d_unew, N*sizeof(float)); 
    cudaMalloc(&d_fout, N*sizeof(float)); 
    cudaMalloc(&dev_partial_c, Grid*Grid*sizeof(float)) ;
   
   //bind texture memory 
   // cudaBindTexture( NULL, tex_fout, d_fout, N*sizeof(float) );
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
          evolve<<<blocksPerGrid, threadsPerBlock>>>(d_fout, d_unew, flag, dx, nc, nr);
        else
          evolve<<<blocksPerGrid, threadsPerBlock>>>(d_fout, d_uold, flag, dx, nc, nr);
        flag = !flag;
        if (count % 500 == 499){//check convergence every 500 steps
        error1<<<blocksPerGrid, threadsPerBlock>>>( dev_partial_c, nc, nr);
        cudaMemcpy( partial_c, dev_partial_c, Grid*Grid*sizeof(float), cudaMemcpyDeviceToHost ) ;
        c = 0;
        for (int i=0; i<Grid*Grid; i++) {
            c += partial_c[i];
        //   std::cout << "error: " << partial_c[i] << std::endl;
         }
       }
      // c = 1;
              
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
    out.open("uend_v3.txt");
    
    for (int i = 0; i < nc; i++) {
        for (int j = 0; j < nr; j++) {
            out << unew[i*nc + j] << ' ';
        }
        out << '\n';
    }


   //free and unbind memory 
    cudaFree( d_uold );
    cudaFree( d_unew );
    cudaFree( d_fout );
    cudaFree( dev_partial_c );
    cudaUnbindTexture( tex_uold );
    cudaUnbindTexture( tex_unew );
 //   cudaUnbindTexture( tex_fout );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    free( unew );
    free( partial_c );
    return 0;
}
