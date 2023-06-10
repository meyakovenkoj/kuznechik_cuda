#include "kuznechik.h"
#include "cpu_kuznechik.h"
#include <stdlib.h>
#include <string.h>

void ctr_encrypt(uint8_t *inparray, 
                uint8_t *outarray,
                uint8_t *key_data,
                const uint32_t numblock)
{
    uint8_t rseed[8] = {0, 5, 1, 0, 0, 0, 6, 24};
	uint8_t *ls_matrix = (uint8_t *)malloc(sizeof(uint8_t) * 256*16*16);
	uint8_t rkey[160];
    create_ls_matrix(ls_matrix, key_data, rkey);
    #ifdef CPU_PROG
	internal_ctr_encrypt(inparray, numblock, ls_matrix, rseed, rkey);
    memcpy(outarray, inparray, numblock*16);
    #else
    encrypt_cuda((const uint8_t *)inparray, *outarray, ls_matrix, rkey, rseed, numblock);
    #endif
    free(ls_matrix);
}
