#ifndef __KUZNECHIK_H
#define __KUZNECHIK_H

#include "common.h"

// void encrypt(uint8_t *block, uint8_t *ls_matrix, uint8_t *rkey);
// void xor_64_2(uint64_t *result, uint64_t *block);
void create_ls_matrix(uint8_t *ls_matrix, uint8_t *key, uint8_t *rkey);
void expand_key(uint8_t *key, uint8_t *extkey);
void log_data(uint8_t *data, const char *prefix);
// void ctr_encrypt(uint8_t *data, uint64_t dataLen, uint8_t *ls_matrix, uint8_t *iv, uint8_t *rkey);

#endif
