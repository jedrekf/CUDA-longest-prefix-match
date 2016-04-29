#ifndef HEADER_BRUTEFORCE
#define HEADER_BRUTEFORCE

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "winsock.h"
#include "structures.h"

__device__ u_char andMask(u_char length){
	switch (length){
	case 1:
		return 0x80;
		break;
	case 2:
		return 0xC0;
		break;
	case 3:
		return 0xE0;
		break;
	case 4:
		return 0xF0;
		break;
	case 5:
		return 0xF8;
		break;
	case 6:
		return 0xFC;
		break;
	case 7:
		return 0xFE;
		break;
	case 8:
		return 0xFF;
		break;
	}
}

// checks given two bytes on given number of bits if they are equal it returns FALSE if they diverge TRUE
__device__ bool isDivergentInBits(u_char *ipByte, u_char *maskByte, u_char *bits){
	
	u_char ip = *ipByte & andMask(*bits);
	u_char mask = *ipByte & andMask(*bits);
	if (ip != mask)
		return false;
	return true;
}

//Semi-Bruteforce algorithm for assigning IPs to Masks by longest prefix match the assignedMasks is a result(it stores just masks but on places that correspond to given array of IPs)
__global__ void bruteforceKernel(u_char *ips, u_char *masks, u_char *assignedMasks, int noIPs, int noMasks){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	u_char bestMask[IPV4M_B] = { 0, 0, 0, 0, 0 };
	u_char *curMask = (u_char*)malloc(IPV4M_B * sizeof(u_char));
	u_char bestMasklength = 0;
	//check each ip with each mask byte by byte
	//ip length= 4 u_char, mask length = 5 u_char
	//each thread should check only 1 IP against all masks
	u_char bits;
	u_char *cur_bits = (u_char*)malloc(sizeof(u_char));
	bool breakLoop = false;
	u_char currentMaskLength;

	if (i < noIPs && (i%IPV4_B == 0))
	{
		for (int j = 0; j < noMasks - IPV4_B; j++){
			if (j%IPV4M_B == 0){ //start of mask
				//set current mask to all 0's for when the writing of important bits ends it exits with 0 at the end mask[122.4.2.13/16] => mask[122.4.0.0/16]
				for (int m = 0; m < IPV4M_B; m++) {
					curMask[m] = 0;
				}
				bits = masks[j + IPV4_B];//get the number of important bits
				curMask[IPV4_B] = bits; //set mask[4]

				for (int p = 0; p < IPV4_B; p++){
					*cur_bits = 8; //bits to check in an IP
					
					//printf("bits= %d\n", bits);
					if (((int)bits - 8) <= 0) {
						*cur_bits = bits; //if in the next iteration the number of bits to check would be smaller eqal than 0 means no more significant bits to check
						breakLoop = true;
						//printf("Loop broken at bits= %d\n", bits);
					}
					else{
						breakLoop = false;
						bits -= 8; //decrease number of bits to still check by the amount of 1 byte
					}
					curMask[p] = masks[j+p];
					//printf("currmask[%d]: %d\n", p, curMask[p]);
					
					if (!isDivergentInBits(&ips[p + i], &curMask[p], cur_bits)){ //if true there is no divergence add to the longest mask
						if (breakLoop){ // we assign the mask only if it's the last set of significant bytes
							currentMaskLength = p * 8 + *cur_bits; //current masks length (if longer the longest prefix matching ins changed to current)
							if (bestMasklength < currentMaskLength){
								//printf("assigning new mask\n");
								for (int assign = 0; assign < IPV4M_B; assign++){
									bestMask[assign] = curMask[assign];
								}
								bestMasklength = currentMaskLength;
							}
						}
					}
					else{ break; }
					if (breakLoop) break;

				}
			}
		}
		//assign the best mask
		for (int p = 0; p < IPV4M_B; p++){
			assignedMasks[((i/4)*5) + p] = bestMask[p];  // i has values 0,4,8,... we i/4 * 5 to get a 0,5,10,... corresponding to mask assigned array
		}
	}
	free(curMask);
	free(cur_bits);
}
//Semi-Bruteforce algorithm for assigning IPs to Masks by longest prefix match the assignedMasks is a result(it stores just masks but on places that correspond to given array of IPs)
void bruteforce(u_char *ips, u_char *masks, u_char *assignedMasks, int noIPBytes, int noMaskBytes){
	u_char *d_ips, *d_masks, *d_assignedMasks;
	int ips_size, masks_size, assignedmasks_size;
	ips_size = noIPBytes * sizeof(u_char);
	masks_size = noMaskBytes * sizeof(u_char);
	assignedmasks_size = NUM_IPS*IPV4M_B * sizeof(u_char);

	cudaMalloc((void **)&d_ips, ips_size);
	cudaMalloc((void **)&d_masks, masks_size);
	cudaMalloc((void **)&d_assignedMasks, assignedmasks_size);

	cudaMemcpy(d_ips, ips, ips_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_masks, masks, masks_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_assignedMasks, assignedMasks, assignedmasks_size, cudaMemcpyHostToDevice);

	bruteforceKernel << <(noIPBytes + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_ips, d_masks, d_assignedMasks, noIPBytes, noMaskBytes);
	cudaMemcpy(assignedMasks, d_assignedMasks, assignedmasks_size, cudaMemcpyDeviceToHost);

	cudaFree(d_ips);
	cudaFree(d_masks);
	cudaFree(d_assignedMasks);
}


#endif