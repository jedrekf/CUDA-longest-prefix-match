#ifndef HEADER_FUNCTIONS
#define HEADER_FUNCTIONS

#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ctype.h"
#include "structures.h"


void init(void);
__global__ void initKernel(void);
u_char countUniqueMaskBytes(u_char *maskByteList);

void createTrie(TrieNode *root, u_char *masks);
void getByteMaskArr(u_char *maskNodeArr, u_char *masks, u_char no_byte);
__global__ void getByteMaskArrKernel(bool *byteMaskArr, u_char **masks, int size, int no_masks, u_char no_byte);
//works after the first byte check
__global__ void getChldMaskArrKernel(MaskNode *chldrn, int size_chldrn, u_char *masks, int size_masks, u_char *mask, u_char byte){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	MaskNode *node;
	node->key = -1;
	int counter = 0;
	int bitsc = 0;
	if (i < size_chldrn)
	{
		for (int j = 0; j < size_masks; j++){
			if (j % (byte + 1) == byte){ j += (IPV4M_B - byte); } //the j is out of what we check in mask skip to eom
			else{
				if (masks[j] != mask[j % IPV4M_B]) break;
				counter++;
				if (counter == byte){ //if yes then the set amount of bits to get here thorugh a mask is fine
					if ((masks[j + (IPV4M_B - byte)] - 8 * counter) > 0){//means the mask reachs here
						node->bits[bitsc] = masks[j + (IPV4M_B - byte)] - 8 * counter;
						bitsc++;
					}
				}
			}
		}

		chldrn[i].key = node->key;
		chldrn[i].bits = node->bits;
	}

}
// byte- byte at which we find children of a mask- mask matching the route
void getchldrn(MaskNode *chldrn, u_char *masks, u_char *mask, u_char byte){

}

//create a trie from given set of masks, one leaf at a time
void createTrie(TrieNode *root, u_char *masks){
	MaskNode *maskNodeArr = (MaskNode*)malloc(U_CHAR_SIZE*sizeof(MaskNode));
	u_char *maskByteList = (u_char *)malloc(U_CHAR_SIZE * sizeof(u_char));
	u_char no_byte = 1;
	u_char child_count=0;
	getByteMaskArr(maskByteList, masks, no_byte);
	child_count = countUniqueMaskBytes(maskByteList);

	//TODO: use a structure to determine how many times does this masks byte occur and save to struct <1,8>
	// so the struct should have a mask byte and an array of alternative msb's
	//make createTree loop itself?
	// create fun similiar to getByteMaskArr but get for actual nodes.

	for (int i = 0; i < child_count; i++){
		//root->children[i] = create_trienode() ;
	}
	
}

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

//u_char *getUniqueMaskBytes;

void getByteMaskArr(u_char *byteMaskArr, u_char *masks, u_char no_byte){
	u_char *d_byteMaskArr;
	u_char *d_masks;
	int u_char_size = U_CHAR_SIZE*sizeof(u_char);
	int masks_size = NUM_MASKS*IPV4M_B*sizeof(u_char);
	cudaMalloc((void **)&d_byteMaskArr, u_char_size);
	cudaMalloc((void **)&d_masks, masks_size);

	cudaMemcpy(d_masks, masks, masks_size,cudaMemcpyHostToDevice);

	getByteMaskArrKernel << <(U_CHAR_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_byteMaskArr, d_masks, U_CHAR_SIZE, NUM_MASKS*IPV4M_B, no_byte);
	cudaMemcpy(byteMaskArr, d_byteMaskArr, u_char_size,cudaMemcpyDeviceToHost);
	
	cudaFree(d_byteMaskArr);
	cudaFree(d_masks);
}

//warmup device with empty kernel for better benchmarks
void init(void){
	initKernel << <1, 1 >> >();
	printf("warmed up!\n");
}
//empty kernel
__global__ void initKernel(void){}



#endif