#ifndef HEADER_TREE
#define HEADER_TREE

#include "winsock.h"


/*
A tree structure definitions
mem for children has to be allocated on create
 source: http://kposkaitis.net/blog/2013/03/09/prefix-tree-implementation-in-c/
 */
//structure for tree
typedef struct TrieNode{
	TrieNode **children;
	//b for bit b=zero || b=one depending on wether the current node is left or righ, null for head
	u_char key;
	//array of important bits may vary depending on a mask
	u_char *bits;
	u_char no_bits;
	u_char no_children;
} TrieNode;


TrieNode *create_trienode(u_char _key, u_char *_bits, u_char _no_bits, u_char _no_children);
void destroy_trienode(TrieNode *node);


// key -value of a current part of IP address
// bits- number of significant bits(from prefix)
// no_children- number of children the node will have
TrieNode *create_trienode(u_char _key, u_char *_bits, u_char _no_bits, u_char _no_children){
	TrieNode *node = (TrieNode*)malloc(sizeof(TrieNode));
	node->no_bits = _no_bits;
	node->bits = (u_char*)malloc(_no_bits * sizeof(u_char));
	for (int i = 0; i < _no_bits;i++)	node->bits[i] = _bits[i];
	node->key = _key;
	node->no_children = _no_children;
	node->children = (TrieNode**)malloc(_no_children * sizeof(TrieNode));
	for (int i = 0; i < _no_children; i++) node->children[i] = NULL;
	return node;
}

void destroy_trienode(struct TrieNode *node){
	if (node == NULL) return;

	for (int i = 0; i < node->no_children; i++)
		destroy_trienode(node->children[i]);

	free(node);
}


#endif