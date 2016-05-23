#include <iostream>
#include <math.h>
#include <stdlib.h> 
#include "consts.h"
#include "matrix.h"

#define TILE_WIDTH 40

//-----------------------------------------------


__global__ void MatrixMult(int m, int n, int k, double *a, double *b, double *c)
{

 int row = threadIdx.y + blockIdx.y*blockDim.y;  
 int col = threadIdx.x + blockIdx.x*blockDim.x;  
 
 if((row < m) && (col < k))
 {
  double temp = 0.0;
  for (int i = 0; i < n; ++i)
  {
   temp += a[row*n+i]*b[col+i*k];
  }
  c[row*k+col] = temp; 
 }

}

//--------------------------------------------------

// Compute C = A * B
__global__ void matrixMultiplySharedMem(double * A, double * B, double * C,
  		       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    __shared__ double ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ double ds_N[TILE_WIDTH][TILE_WIDTH];
    int bx = blockIdx.x, by = blockIdx.y,
       tx = threadIdx.x, ty = threadIdx.y,
       Row = by * TILE_WIDTH + ty,
       Col = bx * TILE_WIDTH + tx;
    double Pvalue = 0;

    for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; ++m) {
       if (Row < numARows && m*TILE_WIDTH+tx < numAColumns)
          ds_M[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH+tx];
       else
          ds_M[ty][tx] = 0;
       if (Col < numBColumns && m*TILE_WIDTH+ty < numBRows)
          ds_N[ty][tx] = B[(m*TILE_WIDTH+ty)*numBColumns+Col];
       else
          ds_N[ty][tx] = 0;

       __syncthreads();
       for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += ds_M[ty][k] * ds_N[k][tx];
       __syncthreads();
    }
    if (Row < numCRows && Col < numCColumns)
       C[Row*numCColumns+Col] = Pvalue;
}

//-------------------------------------------------

void matmul_gpu(int n1, int n2, int n3, double *A, double *B, double *C)
{

 double *dev_a, *dev_b, *dev_c;
 
 dim3 dimGrid((n3-1)/TILE_WIDTH+1,(n1-1)/TILE_WIDTH+1,1);
 dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);


 cudaMalloc((void**)&dev_a, n1*n2*sizeof(double));
 cudaMalloc((void**)&dev_b, n2*n3*sizeof(double));
 cudaMalloc((void**)&dev_c, n1*n3*sizeof(double));

 cudaMemcpy(dev_a, A, n1*n2*sizeof(double), cudaMemcpyHostToDevice);
 cudaMemcpy(dev_b, B, n2*n3*sizeof(double), cudaMemcpyHostToDevice);

 // global memory version
 //MatrixMult<<<dimGrid,dimBlock>>>(n1,n2,n3,dev_a,dev_b,dev_c);

 // shared memory version
 matrixMultiplySharedMem<<<dimGrid,dimBlock>>>(dev_a,dev_b,dev_c,n1,n2,n2,n3,n1,n3);

 cudaMemcpy(C, dev_c, n1*n3*sizeof(double), cudaMemcpyDeviceToHost);


 cudaFree(dev_a);
 cudaFree(dev_b);
 cudaFree(dev_c);

}


//----------------------------------------------

