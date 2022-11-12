#include <stdio.h>
#include <stdint.h>
extern "C"
{
#include "cuda_kuznechik.h"
}
#include "common.h"
#include "cuda_helper.h"

__constant__ uint4 d_rkey[10];

inline __host__ __device__ uint4 operator+(uint4 a, unsigned int b)
{
    return make_uint4(a.x + b, a.y + b, a.z + b,  a.w + b);
}

inline __host__ __device__ void operator^=(uint4 &a, uint4 b)
{
    a.x ^= b.x;
    a.y ^= b.y;
    a.z ^= b.z;
    a.w ^= b.w;
}

__global__ void ctr_encrypt(uint4 *data, const uint4 *ls_matrix, uint4 iv, unsigned int numblock)
{
    unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index >= numblock)
        return;
    __shared__ uint4 sd_rkey[10];
    if (threadIdx.x == 0){
        memcpy(sd_rkey, d_rkey, 10 * sizeof(uint4));
    }
    __syncthreads();

	uint4 ctr = iv;
    ctr.x += index;

	uint4 cur_result = make_uint4(0, 0, 0, 0);
	for (int i = 0; i < 9; ++i)
	{
		ctr ^= __ldg(sd_rkey + i);
        cur_result = __ldg(ls_matrix + 16*( ctr.x & 0x000000ff)        + 0);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.x & 0x0000ff00) >> 8)  + 1);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.x & 0x00ff0000) >> 16) + 2);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.x & 0xff000000) >> 24) + 3);
        cur_result ^= __ldg(ls_matrix + 16*( ctr.y & 0x000000ff)        + 4);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.y & 0x0000ff00) >> 8)  + 5);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.y & 0x00ff0000) >> 16) + 6);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.y & 0xff000000) >> 24) + 7);
        cur_result ^= __ldg(ls_matrix + 16*( ctr.z & 0x000000ff)        + 8);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.z & 0x0000ff00) >> 8)  + 9);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.z & 0x00ff0000) >> 16) + 10);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.z & 0xff000000) >> 24) + 11);
        cur_result ^= __ldg(ls_matrix + 16*( ctr.w & 0x000000ff)        + 12);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.w & 0x0000ff00) >> 8)  + 13);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.w & 0x00ff0000) >> 16) + 14);
        cur_result ^= __ldg(ls_matrix + 16*((ctr.w & 0xff000000) >> 24) + 15);
        ctr = cur_result;
	}
	ctr ^= __ldg(sd_rkey + 9);
    data[index] ^= ctr;
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
        uint4 *ddata;
        uint4 *dev_ls_matrix;
        const uint32_t *iv_vec = (const uint32_t *)iv;
        uint4 d_iv = make_uint4(0, 0, iv_vec[0], iv_vec[1]);
        cudaMalloc(&ddata, sizeof(uint4) * current_device->count_blocks);
        CUDA_ERROR();
        cudaMalloc(&dev_ls_matrix, sizeof(uint4) * 256 * 16);
        CUDA_ERROR();
        cudaMemcpyAsync(ddata, block_array, sizeof(uint8_t) * num_uint8_ts, cudaMemcpyHostToDevice);
        CUDA_ERROR();
        cudaMemcpyToSymbolAsync(d_rkey, rkey, sizeof(uint8_t) * 160);
        CUDA_ERROR();
        cudaMemcpyAsync(dev_ls_matrix, ls_matrix, sizeof(uint8_t) * 256 * 16 * 16, cudaMemcpyHostToDevice);
        CUDA_ERROR();

        cudaEventRecord(start);

        ctr_encrypt<<<current_device->BlocksPerGrid, current_device->ThreadsPerBlock>>>(ddata, dev_ls_matrix, d_iv, current_device->count_blocks);
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
        cudaDeviceReset();
        free(config.dev_props);
        free(config.props);

    }
    printf("Time elapsed (Encryption) : %f millisec\n", elapsedTime);
    printf("Encryption speed: %f MB/s\n", (numblock * 16 / 1024 / 1024) / (elapsedTime / 1000));
}
