#ifndef HEADER_TREE_CREATOR
#define HEADER_TREE_CREATOR

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "tree.h"
#include "structures.h"


__device__ unsigned int getByteMask(u_char byte_number);
__global__ void childrenArray(u_char* chldrn_arr, unsigned int *masks, unsigned int masks_number, u_char byte_number);
u_char childrenCount(u_char *chldrnArr);
__global__ void byteArray(u_char *ba, unsigned int *masks, int *split, int split_size, u_char byte_number);
__global__ void calc_separate_chldrn(u_char *separate_chldrn_arr, unsigned int *masks, int *split, int split_size, u_char byte_number);
void createTree(TreeNode *root, unsigned int *masks, int masks_size);


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

__global__ void childrenArrayImproved(u_char* chldrn_arr, unsigned int *masks, unsigned int masks_number, u_char byte_number){
	unsigned int i, byte_mask, current_mask, offset;
	i = blockDim.x * blockIdx.x + threadIdx.x;

	byte_mask = getByteMask(byte_number);
	offset = ((sizeof(int) - byte_number) * 8);

	if (i < masks_number){
		current_mask = masks[i] & byte_mask;
		current_mask = current_mask >> offset;

		chldrn_arr[current_mask] = 1;
		printf("%d\n", current_mask);
	}

}

// sets up a children array of size 256 of{0,1} such that an index corresponds to a value of a mask [0, 0, 0, 3, 4, 0, 6, 0, 0, 0, 11, ...]
__global__ void childrenArray(u_char* chldrn_arr, unsigned int *masks, unsigned int masks_number, u_char byte_number){
	unsigned int i, byte_mask, current_mask, offset;
	i = blockDim.x * blockIdx.x + threadIdx.x;


	byte_mask = getByteMask(byte_number);
	offset = ((sizeof(int) - byte_number) * 8);

	if (i < masks_number){
		i = i << 1;
		current_mask = masks[i] & byte_mask;
		current_mask = current_mask >> offset;

		chldrn_arr[current_mask] = 1;
		printf("%d\n", current_mask);
	}
}

// Count number of children to be created for current address Byte
u_char childrenCount(u_char *chldrnArr){
	int i;
	u_char counter = 0;

	for (i = 0; i < U_CHAR_SIZE; i++){
		if (chldrnArr[i] == 1)
			counter++;
	}
	return counter;
}

__global__ void createByteArrayFirst(u_char *ba, unsigned int *masks, int masks_size, int *split, int split_size, u_char byte_number){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int tempi, byte_mask;
	u_char temp, offset;
	int j = blockIdx.x;
	int k = threadIdx.x;

	if (i < masks_size){
		if (i % 2){
			byte_mask = getByteMask(byte_number);
			offset = (sizeof(int) - byte_number) * 8;
		}
	}


}
//run a block for each child, so that there is #counter threads to calc each bits and masks (problem with lock -> result in synchronous)
__global__ void byteArrayImproved(u_char *baBytes, u_char *baBits, u_char *baEoms,unsigned int *masks, u_char *maskPrefixes, int *splitIdxs, int *splitCounters, u_char childrenCount, u_char byteNumber){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int byte_mask, offset;
	u_char new_val, temp;
	byte_mask = getByteMask(byteNumber);
	offset = (sizeof(int) - byteNumber) * 8;
	if (i < (int)childrenCount){
		byte_mask = (masks[splitIdxs[i]] & byte_mask) >> offset; //byte_mask should be the same for the whole loop (since children are sorted)
		baBytes[i] = byte_mask;
		for (int j = splitIdxs[i]; j < (splitIdxs[i] + splitCounters[i]); j++){//have to be calculated on a single thread because the BITS would be overwritten
			if (maskPrefixes[j] < 8){
				offset = (int)maskPrefixes[j];
				temp = baBits[i];
				new_val = ((temp >> offset) | 1) << offset; //move to right, put a 1 representing the ending prefix, move back and join with previous BITS 
				baBits[i] = baBits[i] | new_val;
				printf("baBits[%d]: %d\n", i, baBits[i]);
				baEoms[i] = 1;
			}
			else{
				maskPrefixes[i] -= 8;
			}
		}
	}
}
// Create ByteArray[] an array such that [key, bits, EndOfMask]  example:  [23, 0100100, 0, ...]
__global__ void byteArray(u_char *ba, unsigned int *masks, int *split, int split_size, u_char byte_number){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int byte_mask, curr_mask, curr_prefix;
	u_char temp, offset;
	if (i < (split_size >> 1)){ //have to check limitations, but ba after this operation should have size of #children
		i = (i << 1);
		ba[i * 3] = split[i];
		for (int j = split[i]; j < (split[i] + ((split[i + 1]) << 1)); j++){ //*2 because size of masks[X/x]
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
			else if (masks[j] == 8){ //here the mask prefix is set to 0 so in the next iteration the node will be marked to eom (no bits added)
				masks[j] -= 8;
			}
			else{
				masks[j] -= 8;
			}
			printf("ba[%d , %d, %d] { %d %d %d }\n", (i * 3), (i * 3 + 1), (i * 3 + 2), ba[j], ba[j + 1], ba[j + 2]);
		}
	}
}
//marks number of children for a given Byte of masks
__global__ void calc_separate_chldrn(u_char *separate_chldrn_arr, unsigned int *masks, int *split, int split_size, u_char byte_number){
	
	unsigned int i, j, k, byte_mask, current_mask, offset, maskIdx;
	i = blockDim.x * blockIdx.x + threadIdx.x; // id of a current thread (g[lobal])
	j = blockIdx.x; // id of a block
	k = threadIdx.x; //id of a thread in a block (starts  from 0 for each block)
	if (j < (split_size >> 1)){
		j = j << 1;
		if (k < split[j+1]){
			k = k << 1; // k has to be multiplied by 2 because masks[mask, prefix]
			//getting a byte_number byte of a mask
			byte_mask = getByteMask(byte_number);
			offset = ((sizeof(int) - byte_number) * 8);
			maskIdx = split[j];
			//getting masks belonging to the same parent tree node (according to split[]) 
			current_mask = masks[maskIdx + k] & byte_mask;
			current_mask = current_mask >> offset;
			masks[maskIdx + k]; //this is the mask that a single thread should determine if it has a child at given point
			
			separate_chldrn_arr[(U_CHAR_SIZE * j) + current_mask] = 1; // mark that a [j-th child of parent node has a #current_mask as a child]
			printf("current mask: %d\n", masks[ maskIdx + k ]);
		}
	}
}

// Function creating a tree, root- of the tree, masks - set of masks, 
void createTreeImproved(TreeNode *root, MaskList maskList, int masks_size){
	int i;
	u_char chldrn_count;
	u_char *chldrn_arr = (u_char*)calloc(U_CHAR_SIZE, sizeof(u_char));

	u_char *d_chldrn_arr;
	unsigned int *d_masks;


	cudaMalloc((void**)&d_chldrn_arr, U_CHAR_SIZE*sizeof(u_char));
	cudaMalloc((void**)&d_masks, masks_size*sizeof(unsigned int));

	cudaMemcpy(d_chldrn_arr, chldrn_arr, U_CHAR_SIZE*sizeof(u_char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_masks, maskList.masks, masks_size*sizeof(unsigned int), cudaMemcpyHostToDevice);

	childrenArrayImproved << <(NUM_MASKS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_chldrn_arr, d_masks, masks_size, 1);
	cudaMemcpy(chldrn_arr, d_chldrn_arr, U_CHAR_SIZE*sizeof(u_char), cudaMemcpyDeviceToHost);

	chldrn_count = childrenCount(chldrn_arr);
	printf("root with : %d children will be created.", chldrn_count);
	cudaFree(d_chldrn_arr);
	
	root = create_treenode(0, 0, 1, chldrn_count); //root of a tree allocated


	//ip split to blocks that can't be concurrently calculated(same ip Byte)
	Split split;
	split.maskIdx = (int*)malloc(chldrn_count*sizeof(int));
	split.count = (int*)malloc(chldrn_count*sizeof(int));
	int counter = 1, j=-1;
	int lastipB = 9999, ipB; //TODO change in next iteration the bytes to 16 and apply a mask
	for (i = 0; i < masks_size; i++){
		ipB = maskList.masks[i] >> 24;
		if (lastipB == ipB){
			counter++;
		}
		else{
			j++;
			split.maskIdx[j] = i;
			counter = 1;
			lastipB = ipB;
		}
		split.count[j] = counter;
		printf("split arr[%d] -> idx: %d count: %d \n", j, split.maskIdx[j], split.count[j]);
	}

	if (++j == chldrn_count){
		printf("correct splitting for blocks.\n");
	}

	ByteArray ba;
	ba.bytes = (u_char*)calloc(chldrn_count, sizeof(u_char));
	ba.bits = (u_char*)calloc(chldrn_count, sizeof(u_char));
	ba.eom = (u_char*)calloc(chldrn_count, sizeof(u_char));

	u_char *d_bytes, *d_bits, *d_eom, *d_prefixes;
	int *d_splitMaskIdxs, *d_splitCounters;
	cudaMalloc((void**)&d_bytes, chldrn_count * sizeof(u_char));
	cudaMalloc((void**)&d_bits, chldrn_count * sizeof(u_char));
	cudaMalloc((void**)&d_eom, chldrn_count * sizeof(u_char));
	cudaMalloc((void**)&d_prefixes, NUM_MASKS * sizeof(u_char));
	cudaMalloc((void**)&d_splitMaskIdxs, chldrn_count * sizeof(int));
	cudaMalloc((void**)&d_splitCounters, chldrn_count * sizeof(int));
	//TODO: copy data and use in algorithm to create byteArray properly;
	//TODO: check if children_count is the same as athe size of the split array should be
	cudaMemcpy(d_prefixes, maskList.prefixes, NUM_MASKS * sizeof(u_char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bits, ba.bits, chldrn_count * sizeof(u_char), cudaMemcpyHostToDevice); //has to be copied because value is taken to store earlier recorded prefixes
	cudaMemcpy(d_eom, ba.eom, chldrn_count * sizeof(u_char), cudaMemcpyHostToDevice); // has to be copied because only eom is marked
	cudaMemcpy(d_splitMaskIdxs, split.maskIdx, chldrn_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitCounters, split.count, chldrn_count * sizeof(int), cudaMemcpyHostToDevice);
	printf("current children count: %d", chldrn_count);
	byteArrayImproved << <(chldrn_count + THREADS_PER_BLOCK -1), THREADS_PER_BLOCK >> >(d_bytes, d_bits, d_eom,
		d_masks, d_prefixes, d_splitMaskIdxs, d_splitCounters, chldrn_count, 1);


	cudaMemcpy(ba.bytes, d_bytes, chldrn_count*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(ba.bits, d_bits, chldrn_count*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(ba.eom, d_eom, chldrn_count*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(maskList.prefixes, d_prefixes, NUM_MASKS * sizeof(u_char), cudaMemcpyDeviceToHost);

	for (i = 0; i < chldrn_count; i++){
		printf("ba[%d] -> {%d, %d, %d}\n", i, ba.bytes[i], ba.bits[i], ba.eom[i]);
	}
	////here run a kernel for each split block to calculate children
	//cudaMalloc((void**)&d_chldrn_arr, chldrn_count*U_CHAR_SIZE*sizeof(u_char));

	//calc_separate_chldrn << <chldrn_count, THREADS_PER_BLOCK >> > (d_chldrn_arr, d_masks, split, split_size, 2);
	//printf("separate children calculated");
	//free(chldrn_arr);
	//chldrn_arr = (u_char*)malloc(chldrn_count*U_CHAR_SIZE*sizeof(u_char));
	//for (i = 0; i< chldrn_count* U_CHAR_SIZE; i++){
	//	chldrn_arr[i] = 0;	
	//}
	//cudaMemcpy(chldrn_arr, d_chldrn_arr, chldrn_count * U_CHAR_SIZE * sizeof(u_char), cudaMemcpyDeviceToHost);
	//cudaFree(d_chldrn_arr);

	////count children for bigger chldrn_arr

	////create a first level of a tree based on ba data
	////for (i = 0; i < chldrn_count*3; i+=3){
	////	root->children[i / 3] = create_treenode(ba[i], ba[i + 1], ba[i + 2], 22);
	////}



	cudaFree(d_masks);
	cudaFree(d_bytes);
	cudaFree(d_bits);
	cudaFree(d_eom);
	cudaFree(d_prefixes);
	cudaFree(d_masks);
	cudaFree(d_splitMaskIdxs);
	cudaFree(d_splitCounters);
	free(ba.bytes);
	free(ba.bits);
	free(ba.eom);
	free(split.maskIdx);
	free(split.count);
	free(chldrn_arr);
}

// Function creating a tree, root- of the tree, masks - set of masks, 
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

	childrenArray << <(NUM_MASKS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (d_chldrn_arr, d_masks, masks_size / 2, 1);
	cudaMemcpy(chldrn_arr, d_chldrn_arr, U_CHAR_SIZE*sizeof(u_char), cudaMemcpyDeviceToHost);

	chldrn_count = childrenCount(chldrn_arr);

	cudaFree(d_chldrn_arr);

	root = create_treenode(0, 0, 1, chldrn_count); //root of a tree allocated


	//ip split to blocks that can't be concurrently calculated(same ip Byte)
	int split_size = 0;
	int *split = (int*)malloc(2 * U_CHAR_SIZE*sizeof(int));
	int counter = 1, j = -2;
	int lastipB = 9999, ipB; //TODO change in next iteration the bytes to 16 and apply a mask
	for (i = 0; i < masks_size; i += 2){
		ipB = masks[i] >> 24;
		if (lastipB == ipB){
			counter++;
		}
		else{
			j += 2;
			split[j] = i / 2;
			counter = 1;
			lastipB = ipB;
		}
		split[j + 1] = counter;
		printf("split arr[%d] -> %d ; ", j, split[j]);
		printf("split arr[%d] -> %d\n", j+1, split[j+1]);
	}

	split_size = j + 2;
	if (split_size / 2 == chldrn_count){
		printf("correct splitting for blocks.\n");
	}

	u_char *ba = (u_char*)calloc(chldrn_count * 3, sizeof(u_char)); //ba [IP_B, 1-8, 0-1, ...]
	u_char *d_ba;
	int *d_split;

	cudaMalloc((void**)&d_ba, chldrn_count * 3 * sizeof(u_char));
	cudaMalloc((void**)&d_split, split_size* sizeof(int));
	cudaMemcpy(d_split, split, split_size * sizeof(int), cudaMemcpyHostToDevice);

	byteArray << < (chldrn_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_ba, d_masks, split, split_size, 1);

	cudaMemcpy(ba, d_ba, chldrn_count * 3 * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(masks, d_masks, masks_size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

	//here run a kernel for each split block to calculate children
	cudaMalloc((void**)&d_chldrn_arr, chldrn_count*U_CHAR_SIZE*sizeof(u_char));

	calc_separate_chldrn << <chldrn_count, THREADS_PER_BLOCK >> > (d_chldrn_arr, d_masks, split, split_size, 2);
	printf("separate children calculated");
	free(chldrn_arr);
	chldrn_arr = (u_char*)malloc(chldrn_count*U_CHAR_SIZE*sizeof(u_char));
	for (i = 0; i< chldrn_count* U_CHAR_SIZE; i++){
		chldrn_arr[i] = 0;
	}
	cudaMemcpy(chldrn_arr, d_chldrn_arr, chldrn_count * U_CHAR_SIZE * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaFree(d_chldrn_arr);

	//count children for bigger chldrn_arr

	//create a first level of a tree based on ba data
	//for (i = 0; i < chldrn_count*3; i+=3){
	//	root->children[i / 3] = create_treenode(ba[i], ba[i + 1], ba[i + 2], 22);
	//}



	cudaFree(d_masks);
	cudaFree(d_split);
	cudaFree(d_ba);

	free(ba);

	printf("created root with %d children\n", chldrn_count);

	free(chldrn_arr);
}


#endif