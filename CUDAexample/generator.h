#pragma once
#include "stdio.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "winsock.h"
#include "ctype.h"
#include "time.h"
#include "functions.h"
#include "stdint.h"
#include "structures.h"
#include "print.h"

int generate_ip_addresses(u_char *ips);
int generate_masks(u_char *masks);

int generate_ip_addresses(u_char *ips){
	srand(time(NULL));
	int r;
	for (int i = 0; i < NUM_IPS*IPV4_B; i++){
		r = rand() % 256;
		ips[i] = r;
	}
	print_ip(ips);
	return 0;
}

int generate_masks(u_char *masks){
	srand(time(NULL));
	int r;
	for (int i = 1; i < NUM_MASKS*IPV4M_B+1; i++){
		if (i%5==0)
			r = (rand() % 32)+1;
		else
			r = rand() % 256;
		masks[i - 1] = r;
	}
	print_mask(masks);
	return 0;
}