#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "functions.h"
#include "tree.h"
#include "generator.h"
#include "bruteforce.h"
#include "tree_creator.h"
#include "print.h"
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <list>
#include <vector>

__global__ void assignMaskValues(unsigned int *keys, int *values, unsigned int *masks, int size){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size){
		if (i % 2){
			values[i/2] = masks[i];
		}
		else{
			keys[i/2] = masks[i];
		}
	}
}

__global__ void assignSortedMaskValues(unsigned int *masks, unsigned int *keys, int *values, int size){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size){
		if (i % 2){
			masks[i] = values[i / 2];
		}
		else{
			masks[i] = keys[i / 2];
		}
	}
}

void sortMasks(unsigned int *masks, int masks_size){
	int i;
	unsigned int *d_keys, *d_masks; int *d_values;
	
	unsigned int *keys = (unsigned int*)malloc((masks_size/2)*sizeof(unsigned int));
	int *values = (int*)malloc((masks_size/2)*sizeof(int));
	
	cudaMalloc((void**)&d_masks, masks_size*sizeof(unsigned int));
	cudaMalloc((void**)&d_keys, (masks_size / 2)*sizeof(unsigned int));
	cudaMalloc((void**)&d_values, (masks_size/2)*sizeof(int));
	
	cudaMemcpy(d_masks, masks, masks_size*sizeof(unsigned int), cudaMemcpyHostToDevice);

	assignMaskValues << <(masks_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_keys, d_values, d_masks, masks_size);
	
	cudaMemcpy(keys, d_keys, (masks_size / 2)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaMemcpy(values, d_values, (masks_size / 2)*sizeof(int), cudaMemcpyDeviceToHost);
	/*---------------- SORT---------------*/
	thrust::sort_by_key(keys, keys + masks_size/2, values);

	cudaMemcpy(d_keys, keys, (masks_size / 2)*sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_values, values, (masks_size / 2)*sizeof(unsigned int), cudaMemcpyHostToDevice);

	assignSortedMaskValues << <(masks_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(d_masks, d_keys, d_values, masks_size);

	cudaMemcpy(masks, d_masks, masks_size*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	free(keys); free(values);
	cudaFree(d_keys); cudaFree(d_masks); cudaFree(d_values);
}

int main()
{ 
	cudaError_t cudaStatus;
	cudaEvent_t start, stop, start_tree, stop_tree;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&start_tree);
	cudaEventCreate(&stop_tree);
	float elapsedTime;
	
	///////////////////////////// INIT IPS AND MASKS ///////////////////////////
	//init ips and masks
	unsigned int *ips = (unsigned int*)malloc(NUM_IPS * sizeof(unsigned int));
	unsigned int *masks = (unsigned int*)malloc(NUM_MASKS * sizeof(unsigned int) * 2);
	unsigned int *assignedMasks = (unsigned int*)malloc(NUM_IPS * sizeof(unsigned int) * 2);

	printf("mem for IPs and MASKs allocated.\n");
	////////////////////////////////////////////////////////////////////////////

	//warmup by empty kernel
	init();

	generate_ip_addresses(ips);
	printf("IPs generated on CPU\n");
	generate_ip_masks(masks);
	printf("Masks generated on CPU\n");
	///////////////////////////// SORT ///////////////////////////////////
	sortMasks(masks, NUM_MASKS*2);
	printf("Masks sorted.");
	////////////////////////////// BRUTE FORCE //////////////////////////////////
	cudaEventRecord(start);
	bruteforce(ips, masks, assignedMasks);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time elapsed for bruteforce: %f\n", elapsedTime);
	writeToFile(ips, assignedMasks);
	//////////////////////////////	TREE /////////////////////////////////

	TreeNode *root = (TreeNode *)malloc(sizeof(TreeNode));
	cudaEventRecord(start_tree);

	createTree(root, masks, NUM_MASKS*2);
	//TODO here assign ips to masks - tree 

	cudaEventRecord(stop_tree);
	cudaEventSynchronize(stop_tree);
	cudaEventElapsedTime(&elapsedTime, start_tree, stop_tree);
	printf("time elapsed for tree search: %f\n", elapsedTime);

	//not creating a tree so whatever
	//destroy_treenode(root);

	free(root);
	free(ips);
	free(masks);
	free(assignedMasks);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

