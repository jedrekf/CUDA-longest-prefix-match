#ifndef HEADER_BRUTEFORCE
#define HEADER_BRUTEFORCE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "winsock.h"
#include "structures.h"


__device__ int check_assign(unsigned int *ip, unsigned int *mask, int sbs){
	unsigned int important_bits_mask = MAX_MASK << (32 - sbs); // this makes from (1111 1111 << 5 ) == 1110 0000

	unsigned int ipb = (*ip & important_bits_mask);
	//*mask = (*mask & important_bits_mask);
	unsigned int maskb = (*mask & important_bits_mask);
	if (ipb == maskb)
		return 1;
	return 0;
}

//Semi-Bruteforce algorithm for assigning IPs to Masks by longest prefix match the assignedMasks is a result(it stores just masks but on places that correspond to given array of IPs)
__global__ void bruteforceKernel(unsigned int *ips, unsigned int *masks, unsigned int *assignedMasks){
	unsigned int i, j, bestMaskLength = 0, currMaskLength, assignedMaskIndex, currMask;
	i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if (i < NUM_IPS){
	
		bestMaskLength = 0;
		assignedMaskIndex = i << 1; //since #assignedMasks is == 2*NUM_IPS | shift left by 1 = *2

		for (j = 0; j<NUM_MASKS * 2; j += 2){

			currMaskLength = masks[j + 1];
			currMask = masks[j];

			if (check_assign(&ips[i], &masks[j], currMaskLength)){ //check if mask is good for an address
				if (currMaskLength > bestMaskLength){ //swap if it's current longest prefix match
					bestMaskLength = currMaskLength;
					assignedMasks[assignedMaskIndex] = currMask;
					assignedMasks[assignedMaskIndex + 1] = currMaskLength;
				}
			}

		}
		if (bestMaskLength == 0){ //if no best mask found assign default gateway 0.0.0.0/0
			assignedMasks[assignedMaskIndex] = 0; assignedMasks[assignedMaskIndex + 1] = 0;
		}
	
	}
}
//Semi-Bruteforce algorithm for assigning IPs to Masks by longest prefix match the assignedMasks is a result(it stores just masks but on places that correspond to given array of IPs)
void bruteforce(unsigned int *ips, unsigned int *masks, unsigned int *assignedMasks){
	unsigned int *d_ips, *d_masks, *d_assignedMasks;
	int ips_size, masks_size, assignedmasks_size;
	ips_size = NUM_IPS * sizeof(unsigned int);
	masks_size = NUM_MASKS * sizeof(unsigned int) * 2;
	assignedmasks_size = NUM_IPS * sizeof(unsigned int) * 2;

	cudaMalloc((void **)&d_ips, ips_size);
	cudaMalloc((void **)&d_masks, masks_size);
	cudaMalloc((void **)&d_assignedMasks, assignedmasks_size);

	cudaMemcpy(d_ips, ips, ips_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_masks, masks, masks_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_assignedMasks, assignedMasks, assignedmasks_size, cudaMemcpyHostToDevice);

	bruteforceKernel << <(NUM_IPS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_ips, d_masks, d_assignedMasks);
	cudaMemcpy(assignedMasks, d_assignedMasks, assignedmasks_size, cudaMemcpyDeviceToHost);

	cudaFree(d_ips);
	cudaFree(d_masks);
	cudaFree(d_assignedMasks);
}


#endif