#ifndef HEADER_GENERATOR
#define HEADER_GENERATOR

#include "time.h"
#include "stdlib.h"
#include "mersenne_twister.h"
#include "bruteforce.h"

void generate_ip_addresses(unsigned int *ips){
    int i, r;
    seedMT(time(NULL)%99 * 3);
    for(i=0; i< NUM_IPS; i++){
        r = randomMT();
        if(r<0)
            r = -r;
        ips[i] = r;
    }
}

void generate_ip_masks(unsigned int *masks){
    int i, r;
    seedMT(time(NULL)%10* 99);
    for(i=0; i< NUM_MASKS*2; i++){
        if(i%2 == 0)
            r = randomMT();
        else
            r = randomMT()%32 + 1;
        if(r < 0 )
            r = -r;
        masks[i] = r;
    }
}

#endif
