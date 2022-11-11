#include <stdio.h>
#include <stdint.h>
extern "C"
{
#include "cuda_kuznechik.h"
}
#include "common.h"
#include "cuda_helper.h"


__device__ void xor_64_2(uint64_t *result, const uint64_t *block)
{
	result[0] ^= block[0];
	result[1] ^= block[1];
}

#define LS_TRANSFORM(i, block) (ls_matrix+((const uint8_t *)block)[i]*256 + i*16)

__device__ void encrypt(uint64_t *block, const uint8_t *ls_matrix, const uint8_t *rkey)
{
	for (int i = 0; i < 9; ++i)
	{
		xor_64_2(block, (const uint64_t *)(rkey + i * 16));
		uint64_t cur_result[2] = {0};
		for (int j = 0; j < 16; j++) {
			const uint64_t *current_block = (const uint64_t *)LS_TRANSFORM(j, block);
			xor_64_2(cur_result, current_block);
		}
		memcpy(block, cur_result, 16);
	}
	xor_64_2(block, (const uint64_t *)(rkey + 9 * 16));
}

__global__ void ctr_encrypt(uint8_t *data, const uint8_t *ls_matrix, const uint8_t *iv, const uint8_t *rkey, const uint64_t numblock)
{
    uint64_t index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= numblock)
        return;

	uint64_t ctr[2];
    ctr[0] = index;
    ctr[1] = ((const uint64_t *)iv)[0];

    encrypt(ctr, ls_matrix, rkey);
    xor_64_2((uint64_t *)(data + 16 * index), (const uint64_t *)ctr);
}

void encrypt_cuda(const uint8_t *inparray, uint8_t *outarray, const uint8_t *ls_matrix, const uint8_t *rkey, const uint8_t *iv, const uint32_t numblock)
{
    struct IntCudaConfig config = {0};
    IntCudaInitAndConfig(&config, numblock);
    printf("block_number: %d\n", numblock);
    float elapsedTime = 0;
    for (int device_id = 0; device_id < config.deviceNum; device_id++)
    {
        cudaSetDevice(device_id);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        printf("devices: %d\n", config.deviceNum);
        struct deviceConfig *current_device = config.dev_props + device_id;
        printf("current_device(all block num in grid): %d\n", current_device->block_num);
        printf("current_device(real blocks of data): %d\n", current_device->count_blocks);
        uint64_t num_uint8_ts = current_device->count_blocks * 16;
        const uint8_t *block_array = inparray + current_device->block_start * BLOCK_SIZE;
        uint8_t *out_block_array = outarray + current_device->block_start * BLOCK_SIZE;
        uint8_t *ddata;
        uint8_t *dev_ls_matrix;
        uint8_t *dev_rkey;
        uint8_t *dev_iv;

        cudaMalloc(&ddata, sizeof(uint8_t) * num_uint8_ts);
        CUDA_ERROR();
        cudaMalloc(&dev_ls_matrix, sizeof(uint8_t) * 256 * 16 * 16);
        CUDA_ERROR();
        cudaMalloc(&dev_rkey, sizeof(uint8_t) * 160);
        CUDA_ERROR();
        cudaMalloc(&dev_iv, sizeof(uint8_t) * 16);
        CUDA_ERROR();
        cudaMemcpyAsync(ddata, block_array, sizeof(uint8_t) * num_uint8_ts, cudaMemcpyHostToDevice);
        CUDA_ERROR();
        cudaMemcpyAsync(dev_rkey, rkey, sizeof(uint8_t) * 160, cudaMemcpyHostToDevice);
        CUDA_ERROR();
        cudaMemcpyAsync(dev_ls_matrix, ls_matrix, sizeof(uint8_t) * 256 * 16 * 16, cudaMemcpyHostToDevice);
        CUDA_ERROR();
        cudaMemcpyAsync(dev_iv, iv, sizeof(uint8_t) * 16, cudaMemcpyHostToDevice);
        CUDA_ERROR();

        cudaEventRecord(start);

        ctr_encrypt<<<current_device->BlocksPerGrid, current_device->ThreadsPerBlock>>>(ddata, dev_ls_matrix, dev_iv, dev_rkey, current_device->count_blocks);
        CUDA_ERROR();

        cudaEventRecord(stop);
        cudaMemcpyAsync(out_block_array, ddata, sizeof(uint8_t) * num_uint8_ts, cudaMemcpyDeviceToHost);
        CUDA_ERROR();

        cudaEventSynchronize(start);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        elapsedTime += milliseconds;
        cudaFree(ddata);
        cudaFree(dev_ls_matrix);
        cudaFree(dev_rkey);
        cudaFree(dev_iv);
        cudaDeviceReset();
        free(config.dev_props);
        free(config.props);

    }
    printf("Time elapsed (Encryption) : %f millisec\n", elapsedTime);
    printf("Encryption speed: %f MB/s\n", (numblock * 16 / 1024 / 1024) / (elapsedTime / 1000));
}
