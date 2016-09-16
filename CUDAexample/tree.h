#ifndef HEADER_TREE
#define HEADER_TREE

#include "winsock.h"
/*
A tree structure definitions
mem for children has to be allocated on create
 source: http://kposkaitis.net/blog/2013/03/09/prefix-tree-implementation-in-c/
 */
//structure for tree
typedef struct TreeNode{
	struct TreeNode **children;
	u_char key;
	//array of important bits stored as 1 byte [0 1 0 1 0 0 0 0] -> [0 2 0 4 0 0 0 0]
	u_char bits;
	u_char eom; //signifies end of mask
	u_char no_children;
} TreeNode;


TreeNode *create_treenode(u_char _key, u_char _bits,u_char _eom, u_char _no_children);
void destroy_treenode(TreeNode *node);


// key -value of a current part of IP address
// bits- number of significant bits(from prefix)
// no_children- number of children the node will have
TreeNode *create_treenode(u_char _key, u_char _bits, u_char _eom, u_char _no_children){
	int i;
	TreeNode *node = (TreeNode*)malloc(sizeof(TreeNode));
	node->bits = _bits;
	node->eom = _eom;
	node->key = _key;
	node->no_children = _no_children;
	node->children = (TreeNode**)malloc(_no_children * sizeof(TreeNode));
	for (i = 0; i < _no_children; i++) node->children[i] = NULL;
	return node;
}

void destroy_treenode(struct TreeNode *node){
	int i;
	if (node == NULL) return;

	for (i = 0; i < node->no_children; i++)
		if (node->children[i] != NULL)
			destroy_treenode(node->children[i]);

	free(node);
}


// give an array of IP's and a root of a tree it's passed to the kernel which for each IP traverses the tree
void traverseTree(unsigned int *ips, TreeNode *root, int masksSize){
	MaskList assignedMasks;
	unsigned int byte_mask;
	int byte_number, offset;
	u_char current_ip_byte;
	unsigned int important_bits_mask;
	assignedMasks.masks = (int*)calloc(masksSize, sizeof(int));
	assignedMasks.prefixes = (u_char*)calloc(masksSize, sizeof(u_char)); //initializing wit 0's because default gateway is 0.0.0.0/0

	/*TreeNode *d_tree;
	cudaMalloc((void**)&d_tree, );

	for (int i = 0; i < root->no_children; i++){
		printf("%d\n", root->children[i]->key);
	}

	byte_number = 1;
	byte_mask = getByteMask(byte_number);
	offset = ((sizeof(int) - byte_number) * 8);
	for (int i = 0; i < NUM_IPS; i++){
		current_ip_byte = ips[i] & byte_mask;
		current_ip_byte = current_mask >> offset;
	}*/
}



#endif
