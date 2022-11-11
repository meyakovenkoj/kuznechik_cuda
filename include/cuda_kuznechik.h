#ifndef __KUZNECHIK_CUDA_H
#define __KUZNECHIK_CUDA_H

#include "common.h"

void encrypt_cuda(const BYTE *inparray, BYTE *outarray, const BYTE *ls_matrix, const BYTE *rkey, const BYTE *iv, const uint32_t numblock);

#endif