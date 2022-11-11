#include "cuda_helper.h"
#include <stdio.h>

void IntCudaInitAndConfig(struct IntCudaConfig *config, int block_number)
{
    cudaGetDeviceCount(&config->deviceNum);
    CUDA_ERROR();
    if (config->deviceNum == 0)
    {
        fprintf(stderr, "There are no available device(s) that support CUDA\n");
        exit(EXIT_FAILURE);
    }
    struct cudaDeviceProp *props = (struct cudaDeviceProp *)malloc(sizeof(struct cudaDeviceProp) * config->deviceNum);
    struct deviceConfig *dev_props = (struct deviceConfig *)malloc(sizeof(struct deviceConfig) * config->deviceNum);
    config->totalSM = 0;
    for (int i = 0; i < config->deviceNum; i++)
    {
        cudaGetDeviceProperties(&props[i], i);
        CUDA_ERROR();
        int num_sm = props[i].multiProcessorCount;
        printf("SM: %d\n", num_sm);
        printf("Threads per sm: %d\n", props[i].maxThreadsPerMultiProcessor);
        printf("Threads per block: %d\n", props[i].maxThreadsPerBlock);
        config->totalSM += num_sm;
        dev_props[i].sm_count = num_sm;
    }

    int start_num = 0;
    int left_blocks = block_number;
    for (int i = 0; i < config->deviceNum; i++)
    {
        float proportion = (float)dev_props[i].sm_count / (float)config->totalSM;
        int current_blocks = (int)(block_number * proportion);
        left_blocks -= current_blocks;
        if (left_blocks < 0)
        {
            current_blocks += (-1)*left_blocks;
            left_blocks = 0;
        }
        int threads_per_block = dev_props[i].sm_count * 32;
        dim3 DimBlock(threads_per_block, 1, 1);
        dim3 DimGrid((current_blocks - 1) / threads_per_block + 1, 1, 1);
        printf("dim grid: %d\ndim block: %d\n", (current_blocks - 1) / threads_per_block + 1, threads_per_block);
        dev_props[i].ThreadsPerBlock = DimBlock;
        dev_props[i].BlocksPerGrid = DimGrid;
        dev_props[i].block_num = ((current_blocks - 1) / threads_per_block + 1) * threads_per_block;
        dev_props[i].block_start = start_num;
        dev_props[i].count_blocks = current_blocks;
        start_num += current_blocks;
    }
    if (left_blocks > 0) {
        dev_props[config->deviceNum-1].count_blocks += left_blocks;
        dev_props[config->deviceNum-1].block_num += left_blocks;
    }
    config->props = props;
    config->dev_props = dev_props;
}