#ifndef HEADER_TREE_CREATOR
#define HEADER_TREE_CREATOR

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tree.h"
#include "winsock.h"
#include "structures.h"

__device__ unsigned int getByteMask(u_char byte_number){
	switch (byte_number){
	case 1:
		return 0xFF000000;
	case 2:
		return 0x00FF0000;
	case 3:
		return 0x0000FF00;
	case 4:
		return 0x000000FF;
	default:
		return 0;
	}
}

// sets up a children array of size 256 of{0,1} such that an index corresponds to a value of a mask [0, 0, 0, 3, 4, 0, 6, 0, 0, 0, 11, ...]
__global__ void childrenArray(u_char* chldrn_arr, unsigned int *masks, unsigned int masks_number, u_char byte_number){
	unsigned int i, byte_mask, current_mask, offset;
	i = blockDim.x * blockIdx.x + threadIdx.x;

	
	byte_mask = getByteMask(byte_number);
	offset = ((sizeof(int) - byte_number) * 8);

	if (i < masks_number){
		i= i << 1;
		current_mask = masks[i] & byte_mask;
		current_mask = current_mask >> offset;

		chldrn_arr[current_mask] = 1;
	}
}

// Count number of children to be created for current address Byte
u_char childrenCount(u_char *chldrnArr){
	int i;
	u_char counter = 0;

	for (i = 0; i < U_CHAR_SIZE; i++){
		if(chldrnArr[i] == 1)
			counter++;
	}
	return counter;
}

//split contains start and size of a separate blocks (having different ip bytes) we want to calculate
__global__ void byteArrayLimited(u_char *ba, unsigned int *masks, int *split, int split_size, u_char byte_number){
	int i = blockDim.x * blockIdx.x + threadIdx.x, j;
	unsigned int byte_mask, curr_mask, curr_prefix;
	u_char temp, offset;
	if (i < split_size / 2){
		i = i * 2 + 1;
		for (j = split[i]; j < (split[i + 1] * 2); j++){ //*2 because size of masks[X/x]
			curr_mask = masks[j]; j++;
			curr_prefix = masks[j];

			if (curr_prefix < 8){ //if prefix at this moment <8 means we mark it on an array as one of endings for this node
				offset = curr_prefix;
				temp = ba[i * 3 + 1];
				u_char newval = (temp >> offset) | 1;
				newval = newval << offset;
				ba[i * 3 + 1] = (ba[i * 3 + 1] | newval);
				ba[i * 3 + 2] = 1;
			}
			else if (masks[j] == 8){
				masks[j] -= 8;
			}
			else{
				masks[j] -= 8;
			}
		}
	}
}

void createTree(TreeNode *root, unsigned int *masks, int masks_size){
	int i;
	u_char chldrn_count;
	u_char *chldrn_arr = (u_char*)malloc(U_CHAR_SIZE*sizeof(u_char));
	for (i = 0; i< U_CHAR_SIZE; i++){
		chldrn_arr[i] = 0;
	}

	u_char *d_chldrn_arr;
	unsigned int *d_masks;

	cudaMalloc((void**)&d_chldrn_arr, U_CHAR_SIZE*sizeof(u_char));
	cudaMalloc((void**)&d_masks, masks_size*sizeof(unsigned int));

	cudaMemcpy(d_chldrn_arr, chldrn_arr, U_CHAR_SIZE*sizeof(u_char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_masks, masks, masks_size*sizeof(unsigned int), cudaMemcpyHostToDevice);

	childrenArray << <(NUM_MASKS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_chldrn_arr, d_masks, masks_size/2, 1);
	cudaMemcpy(chldrn_arr, d_chldrn_arr, U_CHAR_SIZE*sizeof(u_char), cudaMemcpyDeviceToHost);

	chldrn_count = childrenCount(chldrn_arr);

	cudaFree(d_chldrn_arr); 

	root = create_treenode(0, 0, 1, chldrn_count);

	//ip split to blocks that can't be concurrently calculated(same ip Byte)
	int split_size = 0;
	int *split = (int*)malloc(2*U_CHAR_SIZE*sizeof(int));
	unsigned int counter=1, j=0;
	u_char lastipB = masks[0] >> 24, ipB;
	for (i = 1; i < masks_size; i += 2){
		ipB = masks[i] >> 24;
		if (lastipB == ipB){
			counter++;
		}
		else{
			split[j] = i;
			split[j + 1] = counter;
			j += 2;
			counter = 1;
		}
		lastipB = ipB;
	}
	split_size = j;
	if (split_size / 2 == chldrn_count){
		printf("correct.\n");
	}

	u_char *ba = (u_char*)malloc(chldrn_count*3*sizeof(u_char)); //ba [IP_B, 1-8, 0-1, ...]
	u_char *d_ba;
	int *d_split;

	cudaMalloc((void**)&d_ba, chldrn_count*3*sizeof(u_char));
	cudaMalloc((void**)&d_split, split_size* sizeof(int));
	//cudaMemcpy(d_masks, masks, masks_size*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_split, split, 2 * U_CHAR_SIZE*sizeof(int), cudaMemcpyHostToDevice);

	byteArrayLimited << < (chldrn_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_ba, d_masks, split, split_size, 1);

	cudaMemcpy(ba, d_ba, chldrn_count * 3 * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(masks, d_masks, masks_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	/*for (i = 0; i < chldrn_count; i++){
		root->children[i]->; 
	}*/

	cudaFree(d_masks);
	cudaFree(d_split);
	cudaFree(d_ba);

	free(ba);

	printf("created root with %d children", chldrn_count);

	free(chldrn_arr);
}



#endif