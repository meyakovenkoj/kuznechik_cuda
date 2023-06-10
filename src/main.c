#include <ctype.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "input.h"
#include "common.h"
#ifdef CPU_PROG
#include "cpu_kuznechik.h"
#else
#include "cuda_kuznechik.h"
#endif
#include "kuznechik.h"

int main(int argc, char **argv)
{
    const char *input = NULL;
    const char *output = NULL;
    const char *keyfile = NULL;
    int c;
    opterr = 0;

    while ((c = getopt(argc, argv, "k:i:o:")) != -1)
    {
        switch (c)
        {
        case 'k':
            keyfile = optarg;
            break;
        case 'i':
            input = optarg;
            break;
        case 'o':
            output = optarg;
            break;
        case '?':
            if (optopt == 'k' || optopt == 'i' || optopt == 'o')
                fprintf(stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            return 1;
        default:
            exit(1);
        }
    }

    if (!keyfile)
    {
        fprintf(stderr, "Error: keyfile is not set\n");
        exit(1);
    }
    if (!input)
    {
        fprintf(stderr, "Error: input is not set\n");
        exit(1);
    }
    if (!output)
    {
        fprintf(stderr, "Error: output is not set\n");
        exit(1);
    }

    printf("Args: input: %s, output: %s, keyfile: %s\n", input, output, keyfile);

    FILE *infile = fopen(input, "rb");
    fseek(infile, 0, SEEK_END);
    uint32_t in_size = ftell(infile);
    fseek(infile, 0, SEEK_SET);

    FILE *key_fd = fopen(keyfile, "rb");
    fseek(key_fd, 0, SEEK_END);
    int key_size = ftell(key_fd);
    fseek(key_fd, 0, SEEK_SET);
    BYTE key_data[32];
    int err = ReadData(key_size, key_data, key_fd);
    if (err) {
        fprintf(stderr, "Failed to read key data\n");
        return EXIT_FAILURE;
    }
    int padding = 0;
    if (in_size % BLOCK_SIZE)
    {
        padding = BLOCK_SIZE - in_size % BLOCK_SIZE;
    }
    in_size += padding;
    BYTE *in_data = (BYTE *)calloc(in_size, sizeof(BYTE));
    BYTE *out_data = (BYTE *)malloc(in_size * sizeof(BYTE));
    BYTE *cur_data = out_data;

    err = ReadData(in_size - padding, in_data, infile);
    if (err) {
        fprintf(stderr, "Failed to read input data\n");
        return EXIT_FAILURE;
    }

    // uint8_t rseed[8] = {0, 5, 1, 0, 0, 0, 6, 24};

	// uint8_t *ls_matrix = (uint8_t *)malloc(sizeof(uint8_t) * 256*16*16);
	// uint8_t rkey[160];
    // create_ls_matrix(ls_matrix, key_data, rkey);
    FILE *outfile = fopen(output, "wb");

    clock_t start, end;
    double cpu_time_used;
    start = clock();
    // cur_data = out_data;
    // #ifdef CPU_PROG
	// internal_ctr_encrypt(in_data, in_size / 16, ls_matrix, rseed, rkey);
    // cur_data = in_data;
    // #else
    // encrypt_cuda(in_data, out_data, ls_matrix, rkey, rseed, in_size / 16);
    // #endif
    ctr_encrypt(in_data, cur_data, key_data, in_size / 16);
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf(" %f sec elapsed, %d MBs processed\n", cpu_time_used, (in_size / 1024 / 1024));
    printf("Encryption speed: %f MB/s\n", (in_size / 1024 / 1024) / cpu_time_used);
    err = WriteData(in_size - padding, cur_data, outfile);
    if (err) {
        fprintf(stderr, "Failed to write output data\n");
        return EXIT_FAILURE;
    }
    fclose(outfile);
    free(in_data);
    free(out_data);
    // free(ls_matrix);
    return EXIT_SUCCESS;
}
