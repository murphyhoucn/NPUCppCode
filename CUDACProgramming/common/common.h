#include <sys/time.h>

#ifndef __COMMON_H

#define __COMMOM_H

#define CHECK(call)                                                                                               \
    {                                                                                                             \
        const cudaError_t error = call;                                                                           \
        if (error != cudaSuccess)                                                                                 \
        {                                                                                                         \
            printf("Error %s: %d, code: %d, reason: %s\n", __FILE__, __LINE__, error, cudaGetErrorString(error)); \
            exit(1);                                                                                              \
        }                                                                                                         \
    }

inline double Seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
#endif // __COMMON_H