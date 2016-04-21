#pragma once
#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "winsock.h"
#include "ctype.h"
#include "structures.h"

void init(void);
void createTries(TrieNode ** tries, u_char **masks);
void getByteMaskArr(u_char *bajtMaskArr, u_char *masks);
void ipsToBin(u_char *ips, u_char *bips);
void binMASKs(u_char *masks /*use a structure here to store [bin,e]*/);

__global__ void initKernel(void);
__global__ void getBajtMaskArrKernel(bool *bajtMaskArr, u_char **masks, int size, int no_masks);
__global__ void changeIpToBin(u_char *ips, u_char *bips, int size);
__device__ void bin(u_char a, u_char *output);

void ipsToBin(u_char *ips, u_char *bips){
	u_char *d_ips, *d_bips;
	int size = NUM_IPS*IPV4_B*sizeof(u_char);
	cudaMalloc((void **)&d_ips, size);
	cudaMalloc((void **)&d_bips, size);

	cudaMemcpy(d_ips, ips, size, cudaMemcpyHostToDevice);
	changeIpToBin << <(NUM_IPS*IPV4_B + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_ips, d_bips, size);
	cudaMemcpy(d_bips, bips, size, cudaMemcpyDeviceToHost);
}

__global__ void changeIpToBin(u_char *ips, u_char *bips, int size){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size){
		bin(ips[i], &bips[i]);
	}
}

__global__ void getBajtMaskArrKernel(u_char *bajtMaskArr, u_char *masks, int size, int no_masks){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < size)
	{
		bajtMaskArr[i] = 0; //set 0 default, marked address as not appearing
		for (int j = 1; j < no_masks+1; j++){ 
			if (j%5==1) //the first byte of each mask
			if (((int)masks[j-1]) == i){ //if the cell corresponding to our thread exists in the mask 1st byte we mark it 1
				bajtMaskArr[i] = 1;
				break;
			}
		}
	}
}

void getByteMaskArr(u_char *bajtMaskArr, u_char *masks){
	u_char *d_bajtMaskArr;
	u_char *d_masks;
	int u_char_size = U_CHAR_SIZE*sizeof(u_char);
	int masks_size = NUM_MASKS*IPV4M_B*sizeof(u_char);
	cudaMalloc((void **)&d_bajtMaskArr, u_char_size);
	cudaMalloc((void **)&d_masks, masks_size);

	cudaMemcpy(d_masks, masks, masks_size,cudaMemcpyHostToDevice);

	getBajtMaskArrKernel << <(U_CHAR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_bajtMaskArr, d_masks, U_CHAR_SIZE, NUM_MASKS*IPV4M_B);
	cudaMemcpy(bajtMaskArr, d_bajtMaskArr, u_char_size,cudaMemcpyDeviceToHost);
	
	cudaFree(d_bajtMaskArr);
	cudaFree(d_masks);
}

//warmup device with empty kernel for better benchmarks
void init(void){
	initKernel << <1, 1 >> >();
	printf("warmed up!\n");
}
//empty kernel
__global__ void initKernel(void){}

__device__ void bin(u_char a, u_char *output){
	int i = 0;
	for (i = 0; i <8; ++i) {
		output[8 - i - 1] = (a >> i) & 1;
	}
}


