#ifndef HEADER_FUNCTIONS
#define HEADER_FUNCTIONS

#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ctype.h"
#include "structures.h"
#include "tree.h"


__global__ void getByteMaskArrKernel(u_char *byteMaskArr, u_char *masks, int size, int no_masks, u_char no_byte){
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < size)
	{
		byteMaskArr[i] = 0; //set 0 default, marked address as not appearing
		for (int j = 1; j < no_masks+1; j++){
			if (j % IPV4M_B == (int)no_byte) //determines which byte of address we check
			if (((int)masks[j-1]) == i){ //if the cell corresponding to our thread exists in the mask's 1st byte we mark it 1
				byteMaskArr[i] = 1;
				break;
			}
		}
	}
}

//count mask bytes of value 1
u_char countUniqueMaskBytes(u_char *maskByteList){
	u_char counter=0;
	for (int i = 0; i < U_CHAR_SIZE; i++){
		if (maskByteList[i] == (u_char)1) counter++;
	}
	return counter;
}

void getByteMaskArr(u_char *byteMaskArr, u_char *masks, u_char no_byte){
	u_char *d_byteMaskArr;
	u_char *d_masks;
	int u_char_size = U_CHAR_SIZE*sizeof(u_char);
	int masks_size = NUM_MASKS*IPV4M_B*sizeof(u_char);
	cudaMalloc((void **)&d_byteMaskArr, u_char_size);
	cudaMalloc((void **)&d_masks, masks_size);

	cudaMemcpy(d_masks, masks, masks_size,cudaMemcpyHostToDevice);

	getByteMaskArrKernel << <(U_CHAR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_byteMaskArr, d_masks, U_CHAR_SIZE, NUM_MASKS*IPV4M_B, no_byte);
	cudaMemcpy(byteMaskArr, d_byteMaskArr, u_char_size, cudaMemcpyDeviceToHost);

	cudaFree(d_byteMaskArr);
	cudaFree(d_masks);
}


////////////////////// WARMUP ///////////////////////////////////
__global__ void initKernel(void){}
//warmup device with empty kernel for better benchmarks
void init(void){
	initKernel << <1, 1 >> >();
	printf("warmed up!\n");
}



#endif
