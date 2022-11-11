#ifndef __COMMON_H
#define __COMMON_H
#include <stdint.h>

#define BYTE uint8_t
#define BLOCK_SIZE 16

enum run_mode
{
    ENCRYPT = 0,
    DECRYPT = 1
};

#endif
