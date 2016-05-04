#ifndef HEADER_GENERATOR
#define HEADER_GENERATOR

#include "time.h"
#include "stdlib.h"
#include "bruteforce.h"

void generate_ip_addresses(unsigned int *ips){
	int i, r;
	srand(time(NULL) % 99 * 3);
	for (i = 0; i< NUM_IPS; i++){
		r = rand() << 16;
		r = r | rand();
		if (r<0)
			r = -r;
		ips[i] = r;
	}
}

void generate_ip_masks(unsigned int *masks){
	int i, r;
	srand(time(NULL) % 10 * 99);
	for (i = 0; i< NUM_MASKS * 2; i++){
		if (i % 2 == 0){
			r = rand() << 16;
			r = r | rand();
		}
		else
			r = rand() % 32 + 1;
		if (r < 0)
			r = -r;
		masks[i] = r;
	}
}

#endif
