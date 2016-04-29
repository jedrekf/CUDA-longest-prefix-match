#include <stdio.h>
#include <stdlib.h>

#include "time.h"
#include "string.h"
#include "stdlib.h"
#include "generator.h"
#include "bruteforce.h"

char *byte_to_binary(int x)
{
    static char b[33];
    b[0] = '\0';

    unsigned int z=0;
    for (z = 2147483648; z > 0; z >>= 1)
    {
        strcat(b, ((x & z) == z) ? "1" : "0");
    }

    return b;
}

void writeToFile(unsigned int *ips, unsigned int *masks){
    FILE *f = fopen("assigned-ips-masks", "wb");
    if(f == NULL){
        printf("Error opening file\n");
        exit(1);
    }
    int i, j;
    for(i=0;i<NUM_IPS;i++){
        j= i<<1;
        fprintf(f, "IP: %s ", byte_to_binary(ips[i]));
        fprintf(f, "Mask %s / %d\n", byte_to_binary(masks[j]), masks[j+1]);
    }
}

int main()
{

    unsigned int *ips = (unsigned int*)malloc(NUM_IPS * sizeof(unsigned int));
    unsigned int *masks = (unsigned int*)malloc(NUM_MASKS * sizeof(unsigned int) *2);
    unsigned int *assignedMasks = (unsigned int*)malloc(NUM_IPS * sizeof(unsigned int)*2);

    generate_ip_addresses(ips);
    generate_ip_masks(masks);

    bruteforce(ips, masks, assignedMasks);

    writeToFile(ips, assignedMasks);


    free(ips);
    free(masks);
    free(assignedMasks);

    return 0;
}
