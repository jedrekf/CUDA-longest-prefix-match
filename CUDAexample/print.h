#ifndef HEADER_PRINT
#define HEADER_PRINT

#include "stdio.h"
#include "ctype.h"
#include "winsock.h"
#include "structures.h"

void print_ip_bin(u_char *ip);
void print_ip(u_char *ips);
void print_mask(u_char *masks);
void printByteMaskArr(u_char *byteMasks);

void print_ip_bin(u_char ipByte){
	int i = 0;
	for (i = 0; i <8; ++i) {
		printf("%d", (ipByte >> i) & 1);
	}
	printf("\n");
}

void print_ip(u_char* ips){
	for (int i = 1; i < NUM_IPS*IPV4_B + 1; i++){
		if (i % IPV4_B == 0)
			printf("%d\n", (int)ips[i - 1]);
		else
			printf("%d.", (int)ips[i - 1]);
	}
}

void print_mask(u_char* masks){
	for (int i = 1; i < NUM_MASKS*IPV4M_B + 1; i++){
		if (i % IPV4M_B == 0)
			printf("/%d\n", (int)masks[i - 1]);
		else
			printf("%d.", (int)masks[i - 1]);
	}
}

void printByteMaskArr(u_char *byteMasks){
	for (int i = 0; i < U_CHAR_SIZE; i++){
		printf("%d,", (int)byteMasks[i]);
	}
}

void printTrieNode(TrieNode *node){
	printf("Number of children of a node: %d\n nodes value: %d\n", node->no_children, node->key);
}

void printAssigned(u_char* ips, u_char* masks){
	int j = 0, i=0;  // we need 2 iterators as we move across 2 different sizes arrays simultaneously 
	for (i = 1; i < NUM_MASKS*IPV4_B + 1; i++){
		j++;
		if (i % IPV4M_B == 0)
			printf("assigned");
		else
			printf("%d.", (int)ips[i - 1]);
	}
}

void wrtiteToFile(u_char *ips, u_char *masks){

	FILE *f = fopen("assignedIP-mask.txt", "wb");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}
	int j = 0;
	for (int i = 0; i < NUM_IPS*IPV4_B; i++){
		fprintf(f, "IP: %d.%d.%d.%d Mask: %d.%d.%d.%d/%d\n", ips[i], ips[i+1], ips[i+2], ips[i+3],
			masks[j],masks[j+1], masks[j+2], masks[j+3], masks[j+4]);
		i += 3;
		j += 5;
	}

	fclose(f);
}
#endif