#include "input.h"
#include <stdio.h>
#include <stdlib.h>

int ReadData(unsigned long size, BYTE *data, FILE *fd)
{
    int bytes_read = fread(data, sizeof(BYTE), size, fd);
    if (bytes_read != size)
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

int WriteData(unsigned long size, BYTE *data, FILE *fd)
{
    int bytes_wrote = fwrite(data, sizeof(BYTE), size, fd);
    if (bytes_wrote != size)
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}