#ifndef HEADER_BRUTEFORCE
#define HEADER_BRUTEFORCE

#define NUM_IPS 500
#define NUM_MASKS 100
#define MAX_MASK 0xFFFFFFFF


/*void bruteforceTry(unsigned int *ips, unsigned int *masks, unsigned int *assignedMasks){
    unsigned int a = 8;
    unsigned int b = 7;
    unsigned int important_bits = 31;
    if(check_assign(&a, &b, important_bits))
        printf("good");
    else
        printf("bad");
}*/

int check_assign(unsigned int *ip, unsigned int *mask, int sbs){
    unsigned int important_bits_mask = MAX_MASK << (32-sbs); // this makes from (1111 1111 << 5 ) == 1110 0000

    unsigned int ipb = (*ip & important_bits_mask);
    //*mask = (*mask & important_bits_mask);
    unsigned int maskb = (*mask & important_bits_mask);
    if(ipb == maskb)
        return 1;
    return 0;
}

void bruteforce(unsigned int *ips, unsigned int *masks, unsigned int *assignedMasks){
    unsigned int i, j, bestMaskLength=0, currMaskLength, assignedMaskIndex, currMask;

    for(i=0; i<NUM_IPS; i++){

        bestMaskLength = 0;
        assignedMaskIndex = i<<1; //since #assignedMasks is == 2*NUM_IPS | shift left by 1 = *2

        for(j=0; j<NUM_MASKS*2; j+=2){

            currMaskLength = masks[j+1];
            currMask = masks[j];

            if(check_assign(&ips[i], &masks[j], currMaskLength)){ //check if mask is good for an address
                if(currMaskLength > bestMaskLength){ //swap if it's current longest prefix match
                    bestMaskLength = currMaskLength;
                    assignedMasks[assignedMaskIndex] = currMask;
                    assignedMasks[assignedMaskIndex+1]= currMaskLength;
                }
            }

        }
        if(bestMaskLength == 0){ //if no best mask found assign default gateway 0.0.0.0/0
            assignedMasks[assignedMaskIndex]=0; assignedMasks[assignedMaskIndex+1]=0;
        }
    }
    printf("Masks assigned!");
}

#endif
