#ifndef __CUDA_HELPER_H
#define __CUDA_HELPER_H

#define CUDA_ERROR()                                                                                                         \
    do                                                                                                                       \
    {                                                                                                                        \
        cudaError_t __err = cudaGetLastError();                                                                              \
        if (__err)                                                                                                           \
        {                                                                                                                    \
            printf("File: %s, Line: %d, error code: %d, error: %s\n", __FILE__, __LINE__, __err, cudaGetErrorString(__err)); \
            exit(__err);                                                                                                     \
        }                                                                                                                    \
    } while (0)

struct deviceConfig
{
    int sm_count;
    int block_num;
    int block_start;
    int count_blocks;
    dim3 BlocksPerGrid;
    dim3 ThreadsPerBlock;
};

struct IntCudaConfig
{
    int deviceNum;
    int totalSM;
    struct deviceConfig *dev_props;
    struct cudaDeviceProp *props;
};

void IntCudaInitAndConfig(struct IntCudaConfig *config, int block_number);

#endif