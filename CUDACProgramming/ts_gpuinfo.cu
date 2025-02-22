# include <stdio.h>
# include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n -> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FALL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }


    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d:\"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("CUDA Driver Version / Runtime Version:       %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("CUDA Capability Major/Minor version number:  %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Total amount of global memory:               %.2f GBytes\n", (float)deviceProp.totalGlobalMem/pow(1024, 3));
    printf("GPU Clock rate:                              %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate*1e-3f,deviceProp.clockRate*1e-6f);
    printf("Memory Bus width:                            %d-bits\n", deviceProp.memoryBusWidth);

    if(deviceProp.l2CacheSize)
        printf("L2 Cache Size:                               %d bytes\n", deviceProp.l2CacheSize);
    
    printf("Max Texture Dimension Size (x,y,z)           1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n", deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf("Max Layered Texture Size (dim) x layers      1D=(%d) x %d, 2D=(%d, %d) x %d \n", deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
    printf("Total amount of constant memory              %lu bytes\n", deviceProp.totalConstMem); // 常量内存
    printf("Total amout of shared memory per block:      %d bytes\n", deviceProp.regsPerBlock);  // 共享内存
    printf("Wrap size:                                   %d\n", deviceProp.warpSize);
    printf("Maximum number of thread per multiprocesser  %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Maximum number of thread per block:          %d\n", deviceProp.maxThreadsPerBlock);
    printf("Maximum size of each dimension of a block:   %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Maximum size of each dimension of a grid:    %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Maximum memory pitch:                        %lu bytes\n", deviceProp.memPitch); // 最大连续线性内存
    exit(EXIT_SUCCESS);
}