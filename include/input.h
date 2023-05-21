#ifndef __INPUT_H
#define __INPUT_H

#include <stdio.h>
#include "common.h"

int ReadData(unsigned long size, BYTE *array, FILE *fd);

int WriteData(unsigned long size, BYTE *array, FILE *fd);

#endif
