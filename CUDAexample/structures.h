#ifndef HEADER_STRUCTURES
#define HEADER_STRUCTURES

#include "stdio.h"
#include "stdint.h"
#include "trie.h"

//1536 max threads
#define THREADS_PER_BLOCK 512
#define NUM_MASKS 100000
#define NUM_IPS 1000000
#define IPV4_B 4
#define IPV4M_B 5
#define U_CHAR_SIZE 256
#define MAX_MASK 0xFFFFFFFF

// structure for holding different mask bits in a same byte
typedef struct MaskNode{
	int key;
	u_char *bits;
} MaskNode;


#endif