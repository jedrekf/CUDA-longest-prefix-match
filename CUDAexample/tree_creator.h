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
	}

}

// this takes the array of children and calculates the number of children for each split block
u_char *childrenCountImproved(u_char *chldrnArr, u_char curr_chldrn_count){
	int i;
	u_char counter;
	u_char *arr = (u_char*)malloc(curr_chldrn_count * sizeof(u_char));
	for (i = 0; i < curr_chldrn_count; i++){
		counter = 0;
		for (int j = 0; j < U_CHAR_SIZE; j++){
			if (chldrnArr[i*U_CHAR_SIZE + j] == 1) counter++;
		}
		arr[i] = counter;
	}

	return arr;
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

//run a block for each child, so that there is #counter threads to calc each bits and masks (problem with lock -> result in synchronous)
__global__ void byteArrayImproved(u_char *baBytes, u_char *baBits, u_char *baEoms,unsigned int *masks, u_char *maskPrefixes, u_char *masksRemoved, int *splitIdxs, int *splitCounters, int childrenCount, u_char byteNumber){
	int i = blockDim.x * blockIdx.x + threadIdx.x, curr_mask, splitIdx, splitCounter;
	int byte_mask, offset;

	u_char new_val, temp;
	byte_mask = getByteMask(byteNumber);
	offset = (sizeof(int) - byteNumber) * 8;
	if (i < (int)childrenCount){
		splitIdx = splitIdxs[i];
		splitCounter = splitCounters[i];
		
		for (int j = splitIdx; j < (splitIdx + splitCounter); j++){//have to be calculated on a single thread because the BITS would be overwritten
			if (!masksRemoved[j]){
				curr_mask = masks[j];
				byte_mask = (curr_mask & byte_mask) >> offset; //byte_mask should be the same for the whole loop (since children are sorted)
				baBytes[i] = byte_mask;
				offset = (int)maskPrefixes[j];
				if (offset < 8){
					masksRemoved[splitIdx + splitCounter] = 1;//marking current masked as removed because it won't be a part of next tree-level
					temp = baBits[i];
					new_val = ((temp >> offset) | 1) << offset; //move to right, put a 1 representing the ending prefix, move back and join with previous BITS 
					baBits[i] = baBits[i] | new_val;
					baEoms[i] = 1;
				}
				else{
					maskPrefixes[i] -= 8;
				}
			}
		}
	}
}

// Marks the children that exist for each split[] block. children[chldrn_count * 256]
__global__ void calculateSeparateChildrenImproved(u_char *children, unsigned int *masks, u_char *masksRemoved, int *splitIdxs, int *splitCounts, u_char prevChildrenCount, u_char byte_number){
	int j = blockIdx.x;
	int k = threadIdx.x;
	int idx, byte_mask, offset;
	unsigned int currentMask;

	u_char currByte;
	if (j < prevChildrenCount){
		idx = splitIdxs[j];
		if (k < splitCounts[j]){
			//if (masksRemoved[idx + k] == 0){  //if a mask is 'removed' we don't count it to children
				byte_mask = getByteMask(byte_number);
				offset = (int)((sizeof(int) - byte_number) * 8);
				//getting masks belonging to the same parent tree node (according to split[]) 
				currentMask = masks[idx + k] & byte_mask;
				currByte = (u_char)(currentMask >> offset); //value of a current Byte of a mask
				children[j*U_CHAR_SIZE + currByte] = 1;
			//}
		}
	}

}
// using K should work because the splitIdxs contains global (const) indexes for masks, we can split the values and calculate based on k and split amount
__global__ void createSplit(int *newSplitIdxs, int *newSplitCounters, unsigned int *masks, int *splitIdxs, int*splitCounters, int childrenCount, u_char byte_number){
	int k, i = blockDim.x * blockIdx.x + threadIdx.x;
	int idx, prevIPB = 9999, currIPB, byte_mask, offset, counter;

	if (i < childrenCount){
		byte_mask = getByteMask(byte_number);
		offset = (int)((sizeof(int) - byte_number) * 8);
		idx = splitIdxs[i];
		counter = 1;
		k = idx - 1;
		for (int j = 0; j < splitCounters[i]; j++){
			currIPB = (masks[idx + j] & byte_mask) >> offset;
			if (currIPB == prevIPB){
				counter++;
			}
			else{ 
				k++;
				newSplitIdxs[k] = idx + j;
				counter = 1;
				prevIPB = currIPB;
			}
			newSplitCounters[k] = counter;
		}
	}
}

// Function creating a tree, root- of the tree, masks - set of masks, 
TreeNode* createTreeImproved(MaskList maskList, int masks_size){
	TreeNode *root = (TreeNode *)malloc(sizeof(TreeNode));
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
		//printf("split arr[%d] -> idx: %d count: %d \n", j, split.maskIdx[j], split.count[j]);
	}

	if (++j == chldrn_count){
		printf("correct splitting for blocks.\n");
	}

	ByteArray ba;
	ba.bytes = (u_char*)calloc(chldrn_count, sizeof(u_char));
	ba.bits = (u_char*)calloc(chldrn_count, sizeof(u_char));
	ba.eom = (u_char*)calloc(chldrn_count, sizeof(u_char));

	u_char *d_bytes, *d_bits, *d_eom, *d_prefixes, *d_masksRemoved;
	int *d_splitMaskIdxs, *d_splitCounters;
	cudaMalloc((void**)&d_bytes, chldrn_count * sizeof(u_char));
	cudaMalloc((void**)&d_bits, chldrn_count * sizeof(u_char));
	cudaMalloc((void**)&d_eom, chldrn_count * sizeof(u_char));
	cudaMalloc((void**)&d_prefixes, NUM_MASKS * sizeof(u_char));
	cudaMalloc((void**)&d_masksRemoved, masks_size*sizeof(u_char));
	cudaMalloc((void**)&d_splitMaskIdxs, chldrn_count * sizeof(int));
	cudaMalloc((void**)&d_splitCounters, chldrn_count * sizeof(int));

	cudaMemcpy(d_prefixes, maskList.prefixes, NUM_MASKS * sizeof(u_char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_masksRemoved, maskList.removed, NUM_MASKS*sizeof(u_char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bits, ba.bits, chldrn_count * sizeof(u_char), cudaMemcpyHostToDevice); //has to be copied because value is taken to store earlier recorded prefixes
	cudaMemcpy(d_eom, ba.eom, chldrn_count * sizeof(u_char), cudaMemcpyHostToDevice); // has to be copied because only eom is marked
	cudaMemcpy(d_splitMaskIdxs, split.maskIdx, chldrn_count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitCounters, split.count, chldrn_count * sizeof(int), cudaMemcpyHostToDevice);
	printf("current children count: %d\n", chldrn_count);
	
	byteArrayImproved << <(chldrn_count + THREADS_PER_BLOCK -1), THREADS_PER_BLOCK >> >(d_bytes, d_bits, d_eom,
		d_masks, d_prefixes, d_masksRemoved, d_splitMaskIdxs, d_splitCounters, (int)chldrn_count, 1);

	cudaMemcpy(ba.bytes, d_bytes, chldrn_count*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(ba.bits, d_bits, chldrn_count*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(ba.eom, d_eom, chldrn_count*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(maskList.prefixes, d_prefixes, NUM_MASKS * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(maskList.removed, d_masksRemoved, NUM_MASKS * sizeof(u_char), cudaMemcpyDeviceToHost);
	//FREEING THE MEMORY
	cudaFree(d_bytes); cudaFree(d_bits); cudaFree(d_eom); 
	/*for (i = 0; i < chldrn_count; i++){
		printf("ba[%d] -> {%d, %d, %d}\n", i, ba.bytes[i], ba.bits[i], ba.eom[i]);
	}*/

	////here run a kernel for each split block to calculate children
	cudaMalloc((void**)&d_chldrn_arr, chldrn_count*U_CHAR_SIZE*sizeof(u_char)); //this size holds all the possible children (at last step if we didn't count children we would need 4GB of memory)
	free(chldrn_arr); //removing the previous children array and allocating new one
	chldrn_arr = (u_char*)calloc(chldrn_count*U_CHAR_SIZE, sizeof(u_char));
	cudaMemcpy(d_chldrn_arr, chldrn_arr, chldrn_count * U_CHAR_SIZE * sizeof(u_char), cudaMemcpyHostToDevice);

	calculateSeparateChildrenImproved << <chldrn_count, THREADS_PER_BLOCK >> >(d_chldrn_arr, d_masks, 
		d_masksRemoved, d_splitMaskIdxs, d_splitCounters, chldrn_count, 2); //calculate using 2-nd byte

	cudaMemcpy(chldrn_arr, d_chldrn_arr, chldrn_count * U_CHAR_SIZE * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaFree(d_chldrn_arr);
	printf("separate children calculated, now counting amount of children per SPLIT[].\n");
	u_char *chldrn_countArr = childrenCountImproved(chldrn_arr, chldrn_count); //chldrn_countArray is the next step in storing children (count fo each split block)

	/*for (i = 0; i < chldrn_count;i++){
		printf("for split[%d] no children: %d\n", i, chldrn_countArr[i]);
	}*/
	printf("Creating second level of a tree.\n");

	int childrenCount = 0; //total amount of children on the next level
	for (i = 0; i < chldrn_count; i++){ 
		childrenCount += chldrn_countArr[i];
		root->children[i] = create_treenode(ba.bytes[i], ba.bits[i], ba.eom[i], chldrn_countArr[i]);
	}
	free(ba.bytes); free(ba.bits); free(ba.eom);

	printf("2nd Level with %d children created.\n", childrenCount);


	Split secondSplit;
	secondSplit.maskIdx = (int*)malloc(childrenCount*sizeof(int));
	secondSplit.count = (int*)malloc(childrenCount*sizeof(int));

	int *d_secondSplitMaskIdxs, *d_secondSplitMaskCounters;
	cudaMalloc((void**)&d_secondSplitMaskIdxs, childrenCount* sizeof(int));
	cudaMalloc((void**)&d_secondSplitMaskCounters, childrenCount* sizeof(int)); 
	// childrenCount is the new value, chldrn_count is old
	createSplit << <(chldrn_count + THREADS_PER_BLOCK - 1), THREADS_PER_BLOCK >> >(d_secondSplitMaskIdxs, d_secondSplitMaskCounters, 
		d_masks, d_splitMaskIdxs, d_splitCounters, chldrn_count, 2);

	cudaMemcpy(secondSplit.maskIdx, d_secondSplitMaskIdxs, childrenCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(secondSplit.count, d_secondSplitMaskCounters, childrenCount * sizeof(int), cudaMemcpyDeviceToHost);
	/*for (i = 0; i < childrenCount; i++){
		printf("splitIdx: %d counter: %d\n", secondSplit.maskIdx[i], secondSplit.count[i]);
	}*/
	printf("calculated the next SPLIT[]\n");

	ba.bytes = (u_char*)calloc(childrenCount, sizeof(u_char));
	ba.bits = (u_char*)calloc(childrenCount, sizeof(u_char));
	ba.eom = (u_char*)calloc(childrenCount, sizeof(u_char));
	cudaFree(d_splitMaskIdxs); cudaFree(d_splitCounters); //freeing memory for calculating the next tree level

	cudaMalloc((void**)&d_bytes, childrenCount * sizeof(u_char));
	cudaMalloc((void**)&d_bits, childrenCount * sizeof(u_char));
	cudaMalloc((void**)&d_eom, childrenCount * sizeof(u_char));
	cudaMalloc((void**)&d_splitMaskIdxs, childrenCount * sizeof(int));
	cudaMalloc((void**)&d_splitCounters, childrenCount* sizeof(int));

	cudaMemcpy(d_bits, ba.bits, childrenCount * sizeof(u_char), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_eom, ba.eom, childrenCount * sizeof(u_char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitMaskIdxs, secondSplit.maskIdx, childrenCount * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitCounters, secondSplit.count, childrenCount * sizeof(int), cudaMemcpyHostToDevice);

	byteArrayImproved << <(childrenCount + THREADS_PER_BLOCK - 1), THREADS_PER_BLOCK >> >(d_bytes, d_bits, d_eom,
		d_masks, d_prefixes, d_masksRemoved, d_splitMaskIdxs, d_splitCounters, childrenCount, 2);

	cudaMemcpy(ba.bytes, d_bytes, childrenCount*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(ba.bits, d_bits, childrenCount*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(ba.eom, d_eom, childrenCount*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(maskList.prefixes, d_prefixes, NUM_MASKS * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(maskList.removed, d_masksRemoved, NUM_MASKS * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaFree(d_bytes); cudaFree(d_bits); cudaFree(d_eom);

	/*for (i = 0; i < childrenCount; i++){
		printf("ba[%d] -> {%d, %d, %d}\n", i, ba.bytes[i], ba.bits[i], ba.eom[i]);
	}*/


	cudaMalloc((void**)&d_chldrn_arr, childrenCount*U_CHAR_SIZE*sizeof(u_char));
	free(chldrn_arr); //removing the previous children array and allocating new one
	chldrn_arr = (u_char*)calloc(childrenCount*U_CHAR_SIZE, sizeof(u_char));
	cudaMemcpy(d_chldrn_arr, chldrn_arr, childrenCount * U_CHAR_SIZE * sizeof(u_char), cudaMemcpyHostToDevice);

	calculateSeparateChildrenImproved << <childrenCount, THREADS_PER_BLOCK >> >(d_chldrn_arr, d_masks,
		d_masksRemoved, d_splitMaskIdxs, d_splitCounters, childrenCount, 3);

	cudaMemcpy(chldrn_arr, d_chldrn_arr, childrenCount * U_CHAR_SIZE * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaFree(d_chldrn_arr);

	printf("Separate children calculated. Now counting total of children for the next level\n");
	free(chldrn_countArr);
	chldrn_countArr = childrenCountImproved(chldrn_arr, childrenCount);
	int k = 0, lastChildrenCount = 0;
	for (i = 0; i < chldrn_count; i++){
		for (int j = 0; j < root->children[i]->no_children; j++){
			lastChildrenCount += chldrn_countArr[k];
			root->children[i]->children[j] = create_treenode(ba.bytes[k], ba.bits[k], ba.eom[k], chldrn_countArr[k]);
			k++;
		}
	}
	printf("3rd level of a tree created.\n");
	printf("total number of children for this level: %d\n", lastChildrenCount);
	free(ba.bytes); free(ba.bits); free(ba.eom);










	Split thirdSplit;
	thirdSplit.maskIdx = (int*)malloc(lastChildrenCount*sizeof(int));
	thirdSplit.count = (int*)malloc(lastChildrenCount*sizeof(int));

	int *d_thirdSplitMaskIdxs, *d_thirdSplitMaskCounters;
	cudaMalloc((void**)&d_thirdSplitMaskIdxs, lastChildrenCount* sizeof(int));
	cudaMalloc((void**)&d_thirdSplitMaskCounters, lastChildrenCount* sizeof(int));
	// childrenCount is the new value, chldrn_count is old
	createSplit << <(childrenCount + THREADS_PER_BLOCK - 1), THREADS_PER_BLOCK >> >(d_thirdSplitMaskIdxs, d_thirdSplitMaskCounters,
		d_masks, d_secondSplitMaskIdxs, d_secondSplitMaskCounters, childrenCount, 3);

	cudaMemcpy(thirdSplit.maskIdx, d_thirdSplitMaskIdxs, lastChildrenCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(thirdSplit.count, d_thirdSplitMaskCounters, lastChildrenCount * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_thirdSplitMaskIdxs); cudaFree(d_thirdSplitMaskCounters);
	/*for (i = 0; i < childrenCount; i++){
	printf("splitIdx: %d counter: %d\n", secondSplit.maskIdx[i], secondSplit.count[i]);
	}*/
	printf("calculated the next SPLIT[]\n");

	ba.bytes = (u_char*)calloc(lastChildrenCount, sizeof(u_char));
	ba.bits = (u_char*)calloc(lastChildrenCount, sizeof(u_char));
	ba.eom = (u_char*)calloc(lastChildrenCount, sizeof(u_char));
	cudaFree(d_splitMaskIdxs); cudaFree(d_splitCounters); //freeing memory for calculating the next tree level

	cudaMalloc((void**)&d_bytes, lastChildrenCount * sizeof(u_char));
	cudaMalloc((void**)&d_bits, lastChildrenCount * sizeof(u_char));
	cudaMalloc((void**)&d_eom, lastChildrenCount * sizeof(u_char));
	cudaMalloc((void**)&d_splitMaskIdxs, lastChildrenCount* sizeof(int));
	cudaMalloc((void**)&d_splitCounters, lastChildrenCount* sizeof(int));

	cudaMemcpy(d_bits, ba.bits, lastChildrenCount * sizeof(u_char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_eom, ba.eom, lastChildrenCount * sizeof(u_char), cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitMaskIdxs, thirdSplit.maskIdx, lastChildrenCount* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_splitCounters, thirdSplit.count, lastChildrenCount * sizeof(int), cudaMemcpyHostToDevice);

	byteArrayImproved << <(lastChildrenCount + THREADS_PER_BLOCK - 1), THREADS_PER_BLOCK >> >(d_bytes, d_bits, d_eom,
		d_masks, d_prefixes, d_masksRemoved, d_splitMaskIdxs, d_splitCounters, lastChildrenCount, 3);

	cudaMemcpy(ba.bytes, d_bytes, lastChildrenCount*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(ba.bits, d_bits, lastChildrenCount*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(ba.eom, d_eom, lastChildrenCount*sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(maskList.prefixes, d_prefixes, NUM_MASKS * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaMemcpy(maskList.removed, d_masksRemoved, NUM_MASKS * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaFree(d_bytes); cudaFree(d_bits); cudaFree(d_eom);

	/*for (i = 0; i < childrenCount; i++){
	printf("ba[%d] -> {%d, %d, %d}\n", i, ba.bytes[i], ba.bits[i], ba.eom[i]);
	}*/


	cudaMalloc((void**)&d_chldrn_arr, lastChildrenCount*U_CHAR_SIZE*sizeof(u_char));
	free(chldrn_arr); //removing the previous children array and allocating new one
	chldrn_arr = (u_char*)calloc(lastChildrenCount*U_CHAR_SIZE, sizeof(u_char));
	cudaMemcpy(d_chldrn_arr, chldrn_arr, lastChildrenCount * U_CHAR_SIZE * sizeof(u_char), cudaMemcpyHostToDevice);

	calculateSeparateChildrenImproved << <lastChildrenCount, THREADS_PER_BLOCK >> >(d_chldrn_arr, d_masks,
		d_masksRemoved, d_splitMaskIdxs, d_splitCounters, lastChildrenCount, 4); /////////////?????????????????

	cudaMemcpy(chldrn_arr, d_chldrn_arr, lastChildrenCount * U_CHAR_SIZE * sizeof(u_char), cudaMemcpyDeviceToHost);
	cudaFree(d_chldrn_arr);

	printf("Separate children calculated. Now counting total of children for the next level\n");
	free(chldrn_countArr);
	chldrn_countArr = childrenCountImproved(chldrn_arr, lastChildrenCount);
	
	k = 0;
	int finChildrenCount = 0;
	for (i = 0; i < chldrn_count; i++){
		for (int j = 0; j < root->children[i]->no_children; j++){
			for (int l = 0; l < root->children[i]->children[j]->no_children; l++){
				finChildrenCount += chldrn_countArr[k];
				root->children[i]->children[j]->children[l] = create_treenode(ba.bytes[k], ba.bits[k], ba.eom[k], chldrn_countArr[k]);
				k++;
			}
		}
	}

	printf("calculated as fuck.\n");
	printf("amount of children on the last level %d\n", finChildrenCount);
	
	
	cudaFree(d_masks);
	cudaFree(d_bytes);
	cudaFree(d_bits);
	cudaFree(d_eom);
	cudaFree(d_prefixes);
	cudaFree(d_masks);
	cudaFree(d_splitMaskIdxs);
	cudaFree(d_splitCounters);
	cudaFree(d_masksRemoved);
	free(ba.bytes);
	free(ba.bits);
	free(ba.eom);
	free(split.maskIdx);
	free(split.count);
	free(chldrn_countArr);

	return root;
}

#endif