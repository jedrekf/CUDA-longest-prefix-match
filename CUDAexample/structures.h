#ifndef HEADER_STRUCTURES
#define HEADER_STRUCTURES
#include "stdio.h"
#include "winsock.h"

//1536 max threads
#define THREADS_PER_BLOCK 1024
#define NUM_MASKS 50
#define NUM_IPS 10000
#define IPV4_B 4
#define IPV4M_B 5
#define U_CHAR_SIZE 256
#define MAX_MASK 0xFFFFFFFF

//struct for holding list of masks and prefixes
typedef struct MaskList{
	int* masks;
	u_char* prefixes;
	u_char* removed;
}MaskList;

typedef struct ByteArray{
	u_char* bytes;
	u_char* bits;
	u_char* eom;
}ByteArrayBlock;

typedef struct Split{
	int* maskIdx;
	int* count;
}Split;
#endif
