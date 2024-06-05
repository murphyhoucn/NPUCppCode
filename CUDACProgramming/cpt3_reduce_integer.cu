#include <cuda_runtime.h>
#include <stdio.h>
#include "common/common.h"

/*
 * This code implements the interleaved and neighbour-paired approaches to parallel reduction in CUDA.
*/

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
    int gpu_sum = 0;
    
    // allocate device memory
    int *d_idate = NULL;
    int *d_odate = NULL;
    cudaMalloc((void **)&d_idate, bytes);
    cudaMalloc((void **)&d_odate, bytes);

    // cpu reduction
    i_start = Seconds();
    unsigned cpu_sum = RecursiveReduce(tmp, array_size);
    i_elaps = Seconds() - i_start;
    printf("cpu reduce elapsed %f ms cpu_sum: %u\n", i_elaps * 1000, cpu_sum);

    // kernel 1: reduce neighboured
    cudaMemcpy(d_idate, h_idate, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchrnize();
    i_start = Seconds()


}