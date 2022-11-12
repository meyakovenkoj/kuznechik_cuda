CFLAGS = -std=c99 -I./include -O3 -Wall -Werror -pedantic
CXXFLAGS = -std=c++11 -I./include -O3
CC = gcc
NVCC = nvcc
NVCCFLAGS = -Wno-deprecated-gpu-targets --use_fast_math -m64 \
		-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 \
		-gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 \
		-gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 \
		-gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 \
		-gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 \
		-gencode arch=compute_86,code=compute_86
LDFLAGS = -L /usr/local/cuda/lib64 -lcudart
EXECUTABLE := main.exe
CPU_TEST := cpu_test.exe
CPU_MAIN := cpu_main.exe
GEN_MAIN := gen.exe
CU_FILES   := cuda_helper.cu cuda_kuznechik.cu
CC_FILES   := main.c input.c kuznechik.c cpu_test.c
SRCDIR=src
OBJDIR=build
OBJS=$(OBJDIR)/input.o $(OBJDIR)/cuda_helper.o $(OBJDIR)/cuda_kuznechik.o $(OBJDIR)/main.o $(OBJDIR)/kuznechik.o
CPU_OBJS=$(OBJDIR)/kuznechik.o $(OBJDIR)/cpu_test.o $(OBJDIR)/cpu_kuznechik.o
CPU_TEST_OBJS=$(OBJDIR)/cpu_kuznechik.o $(OBJDIR)/kuznechik.o $(OBJDIR)/cpu_main.o $(OBJDIR)/input.o
GEN_SRC=$(SRCDIR)/generator.c $(SRCDIR)/input.c 

.PHONY: all dirs clean

all: $(EXECUTABLE) $(CPU_MAIN)

dirs:
	mkdir -p $(OBJDIR)/

$(EXECUTABLE): dirs $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $< $(CFLAGS) -c -o $@

$(OBJDIR)/cpu_main.o: $(SRCDIR)/main.c
	$(CC) $< $(CFLAGS) -DCPU_PROG -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $< $(CXXFLAGS) $(NVCCFLAGS) -c -o $@

$(CPU_TEST): $(CPU_OBJS)
	$(CC) -o $@ $(CPU_OBJS)

$(CPU_MAIN): $(CPU_TEST_OBJS)
	$(CC) -o $@ $(CPU_TEST_OBJS)

$(GEN_MAIN): $(SRCDIR)/generator.c $(SRCDIR)/input.c 
	$(CC) $(CFLAGS) $(GEN_SRC) -o $@

clean:
	rm -f build/* $(EXECUTABLE) $(CPU_MAIN) $(CPU_TEST) $(GEN_MAIN)