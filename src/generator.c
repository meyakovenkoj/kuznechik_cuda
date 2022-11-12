#include "input.h"
#include <ctype.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

enum BYTE_MOD
{
    KILOBYTE = 0,
    MEGABYTE
};

unsigned int g_seed = 0x10101010;

void gen_rand(uint8_t *a, int count)
{
    unsigned int seed = g_seed;
    for (int i = 0; i < count; i++) {
        seed = (214013 * seed + 2531011);
        a[i] = seed & 0xff;
    }
    g_seed = seed;
}

int main(int argc, char **argv)
{
    int length = 0;
    int size = 0;
    int byte_mod = MEGABYTE;
    int c;

    opterr = 0;

    while ((c = getopt(argc, argv, "kl:s:")) != -1)
        switch (c) {
        case 'k':
            byte_mod = KILOBYTE;
            break;
        case 'l':
            length = atoi(optarg);
            break;
        case 's':
            size = atoi(optarg);
            break;
        case '?':
            if (optopt == 'l' || optopt == 's')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            return 1;
        default:
            abort();
        }

    printf("length = %d, size = %d\n", length, size);

    int multiplyer = 1;
    const char *suffix;
    switch (byte_mod) {
    case MEGABYTE:
        multiplyer *= 1024 * 1024;
        suffix = "mb";
        break;
    case KILOBYTE:
        multiplyer *= 1024;
        suffix = "kb";
        break;
    default:
        suffix = "mb";
        break;
    }
    size_t real_amount = size * multiplyer / sizeof(uint8_t);
    size_t real_count = real_amount / length;
    real_amount = real_count * length;
    FILE *finput;
    char name[100];
    sprintf(name, "gen_%04d%s_1.txt", size, suffix);

    if ((finput = fopen(name, "wb")) == NULL) {
        printf("\nError opening file\n");
        return -1;
    }

    uint8_t *array = malloc(length * sizeof(uint8_t));
    for (int n = 0; n < real_count; n++) {
        gen_rand(array, length);
        WriteData(length, array, finput);
    }
    printf("Total amount = %zu, line size = %d, row number = %zu\n", real_amount, length, real_count);
    free(array);
    fclose(finput);
    return 0;
}