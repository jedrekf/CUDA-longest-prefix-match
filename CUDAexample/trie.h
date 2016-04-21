#pragma once

/*
A tree structure definitions
mem for children has to be allocated on create
 source: http://kposkaitis.net/blog/2013/03/09/prefix-tree-implementation-in-c/
 */

//defining nullable boolean
typedef enum {
	zero,
	one,
	null
} Boolean;

//structure for tree
typedef struct TrieNode {
	struct TrieNode *left; //left = 0
	struct TrieNode *right; //right = 1
	Boolean b; //b for bit b=zero || b=one depending on wether the current node is left or righ, null for head
	bool end_mask;
} TrieNode;

