#pragma once
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
		if (i % 4 == 0)
			printf("%d\n", (int)ips[i - 1]);
		else
			printf("%d.", (int)ips[i - 1]);
	}
}

void print_mask(u_char* masks){
	for (int i = 1; i < NUM_MASKS*IPV4M_B + 1; i++){
		if (i % 5 == 0)
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
