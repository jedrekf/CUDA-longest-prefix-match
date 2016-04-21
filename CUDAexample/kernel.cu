#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "functions.h"
#include "trie.h"
#include "generator.h"


int main()
{ 
	
	cudaError_t cudaStatus;
	//time structures
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	///////////////////////////// INIT IPS AND MASKS ///////////////////////////
	//init ips and masks
	u_char *ips= (u_char*)malloc(NUM_IPS*IPV4_B*sizeof(u_char));
	u_char *masks = (u_char*)malloc(NUM_MASKS*IPV4M_B* sizeof(u_char));
	u_char *bips = (u_char*)malloc(NUM_IPS*IPV4_B*sizeof(u_char));

	printf("mem for IPs and MASKs allocated.\n");
	////////////////////////////////////////////////////////////////////////////

	//warmup by empty kernel
	init();
	cudaEventRecord(start, 0);
	generate_ip_addresses(ips);
	generate_masks(masks);
	//Array for determining the first byte of mask (limit tree nodes)
	u_char *byteMaskArr = (u_char*)malloc(U_CHAR_SIZE*sizeof(u_char));
	getByteMaskArr(byteMaskArr, masks); //sets up array of used ipmasks (their first Byte)

	ipsToBin(ips, bips);
	print_ip_bin(ips[0]);
	//count and take values of nodes from this
	printByteMaskArr(byteMaskArr); 
	//createTries(tries, masks);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("time elapsed for kernel and creating ips/masks: %f\n", &time);

	free(ips);
	free(masks);
	free(byteMaskArr);
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

