#include "cpu_kuznechik.h"
#include <memory.h>
#include <stdio.h>

void xor_64_2(uint64_t *result, uint64_t *block)
{
	result[0] ^= block[0];
	result[1] ^= block[1];
}

void gostencrypt(uint8_t *block, uint8_t *ls_matrix, uint8_t *rkey)
{
	for (int i = 0; i < 9; ++i)
	{
		xor_64_2((uint64_t *)block, (uint64_t *)(rkey + i * 16));
		uint8_t cur_result[16] = {0};
		for (int j = 0; j < 16; j++) {
			uint8_t *current_block = ls_matrix+block[j]*256 + j*16;
			xor_64_2((uint64_t *)cur_result, (uint64_t *)current_block);
		}
		memcpy(block, cur_result, 16);
	}
	xor_64_2((uint64_t *)block, (uint64_t *)(rkey + 9 * 16));
}

void ctr_encrypt(uint8_t *data, uint64_t dataLen, uint8_t *ls_matrix, uint8_t *iv, uint8_t *rkey)
{
	uint64_t ctr[2];
    ctr[0] = 0;
    ctr[1] = ((uint64_t *)iv)[0];
    printf("blocks %llu\n", dataLen);
	for (uint64_t i = 0; i < dataLen; i++) {
		ctr[0] = i;
        ctr[1] = ((uint64_t *)iv)[0];
		gostencrypt((BYTE *)ctr, ls_matrix, rkey);
    	xor_64_2((uint64_t *)(data + 16 * i), ctr);
	}
}
