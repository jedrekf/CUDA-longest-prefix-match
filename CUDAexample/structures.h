#pragma once
#include "stdio.h"
#include "stdint.h"
#include "trie.h"

//1536 max threads
#define THREADS_PER_BLOCK 512
#define NUM_MASKS 10
#define NUM_IPS 20
#define IPV4_B 4
#define IPV4M_B 5
#define U_CHAR_SIZE 256

struct TrieNode *create_trienode(Boolean _b);
void destroy_trienode(struct TrieNode *node);

TrieNode *create_trienode(Boolean _b){
	TrieNode *node = (TrieNode*)malloc(sizeof(TrieNode));
	node->b = _b;
	node->end_mask = false;
	node->left = (TrieNode*)malloc(sizeof(TrieNode));
	node->right = (TrieNode*)malloc(sizeof(TrieNode));
	return node;
}

void destroy_trienode(struct TrieNode *node){
	if (node == NULL) return;
	destroy_trienode(node->left);
	destroy_trienode(node->right);
	
	free(node);
}