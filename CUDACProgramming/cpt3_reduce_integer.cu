#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

/*
 * This code implements the interleaved and neighbour-paired approaches to parallel reduction in CUDA.
*/

// CPU Reduction
unsigned RecursiveReduce(int *data, int const kSize)
{
    // terminate check
    if (kSize == 1) return(data[0]);

    // renew the stride
    int const kStride = kSize / 2;
    
    // in-place reduction
    for (int i = 0; i < kStride; ++i)
    {
        data[i] += data[i + kStride];
    }

    // call recursicely
    return RecursiveReduce(data, kStride);
}

// Neighboured pair implementation with divergence
__global__ void ReduceNeighboured(int *g_idata, int *g_odata, unsigned int kN)
{
    unsigned int t_id = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // define idata as a pointer
    // 
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= kN) return;
    
    for(int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if((t_id % (2 * stride)) == 0)
        {
           idata[t_id] += idata[t_id + stride]; 
        }

        __syncthreads();
    }

    if(t_id == 0) g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, dev);
    printf("%s starting reduction at ", argv[0]);
    printf("device %d : %s ", dev, device_prop.name);
    cudaSetDevice(dev);

    bool b_result = false;

    // initialization
    int array_size = 1<<24;
    printf("with array size %d \n", array_size);

    // execution configuration
    int block_size = 512;
    if(argc > 1){block_size = atoi(argv[1]);}

    dim3 block (block_size, 1);
    dim3 grid ((array_size + block.x - 1) / block.x, 1);
    printf("grid %d, block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = array_size * sizeof(int);
    int *h_idate = (int *)malloc(bytes);
    int *h_odate = (int *)malloc(grid.x * sizeof(int));
    int *tmp = (int *)malloc(bytes);

    // initialization the array
    for (int i = 0; i < array_size; ++i)
    {
        h_idate[i] = (int)(rand() & 0xFF); // 返回一个范围在 0 到 255 之间的整数
    }
    memcpy(tmp, h_idate, bytes);

    double i_start, i_elaps;
    unsigned gpu_sum = 0;
    
    // allocate device memory
    int *d_idate = NULL;
    int *d_odate = NULL;
    CHECK(cudaMalloc((void **)&d_idate, bytes));
    CHECK(cudaMalloc((void **)&d_odate, bytes));

    // cpu reduction
    i_start = Seconds();
    unsigned cpu_sum = RecursiveReduce(tmp, array_size);
    i_elaps = Seconds() - i_start;
    printf("cpu reduce elapsed %f ms. cpu_sum: %u\n", i_elaps * 1000, cpu_sum);

    // kernel 1: reduce neighboured
    CHECK(cudaMemcpy(d_idate, h_idate, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    i_start = Seconds();
    ReduceNeighboured<<<grid, block>>>(d_idate, d_odate, array_size);
    CHECK(cudaDeviceSynchronize());
    i_elaps = Seconds() - i_start;
    CHECK(cudaMemcpy(h_odate, d_odate, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for(int i = 0; i < grid.x; ++i) gpu_sum += h_odate[i];

    printf("gpu neighboured elapsed %f ms. gpu_sum = %u. <<<grid: %d, block:%d>>>. \n", i_elaps * 1000, gpu_sum, grid.x, block.x);

}