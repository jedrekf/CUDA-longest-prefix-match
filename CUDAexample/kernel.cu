#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "functions.h"
#include "trie.h"
#include "generator.h"
#include "bruteforce.h"
#include "print.h"

int main()
{ 
	
	cudaError_t cudaStatus;
	//time structures
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
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
	printf("IPs generated on CPU");
	generate_ip_masks(masks);
	printf("Masks generated on CPU");
	////////////////////////////// BRUTE FORCE //////////////////////////////////
	cudaEventRecord(start);
	bruteforce(ips, masks, assignedMasks);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("time elapsed: %f\n", elapsedTime);
	writeToFile(ips, assignedMasks);
	////////////////////////////////////////////////////////////////////////////

	//Array for determining the first byte of mask (limit tree nodes)
	/*u_char *byteMaskArr = (u_char*)malloc(U_CHAR_SIZE*sizeof(u_char));
	getByteMaskArr(byteMaskArr, masks, 1); //sets up array of used ipmasks (their first Byte)
	u_char no_children_node = countUniqueMaskBytes(byteMaskArr);

	TrieNode *root = (TrieNode *)malloc(sizeof(TrieNode));
	root = create_trienode(0, 0, 0,no_children_node);
	createTrie(root, masks);
	//here assign ips to masks - tree traversing
	destroy_trienode(root);
	
	//count and take values of nodes from this
	printByteMaskArr(byteMaskArr);*/

	free(ips);
	free(masks);
	//free(byteMaskArr);
	free(assignedMasks);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

