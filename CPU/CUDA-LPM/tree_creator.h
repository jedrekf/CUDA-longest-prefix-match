#ifndef HEADER_TREE_CREATOR
#define HEADER_TREE_CREATOR

#include "trie.h"

#define INT_SIZE 32
#define BYTE_MAX 255

unsigned int getByteMask(u_char byte_number){
    switch(byte_number){
        case 1:
            return 0xFF000000;
            break;
        case 2:
            return 0x00FF0000;
            break;
        case 3:
            return 0x0000FF00;
            break;
        case 4:
            return 0x000000FF;
            break;
        default:
            return -1;
            break;
    }


}

void bubbleSort( unsigned int *masks, unsigned int masks_size, u_char byte_number){
    unsigned int temp, byte_mask, offset, current_mask, next_mask;
    byte_mask = getByteMask(byte_number);
    offset =  (INT_SIZE - ((sizeof(int) - byte_number)*8));
    int x, y;
    for(x=0; x<masks_size; x+=2)
	{

		for(y=0; y<masks_size-2; y+=2)

		{
            current_mask = masks[y] & byte_mask;
            current_mask = current_mask >> offset;
            next_mask = masks[y+2] & byte_mask;
            next_mask = next_mask >> offset;
			if(current_mask>next_mask)
			{

				temp = masks[y+2];

				masks[y+2] = masks[y];

				masks[y] = temp;

			}

		}

	}

}

// sets up a children array of size 256 of{0,1} such that an index corresponds to a value of a mask [0, 0, 0, 3, 4, 0, 6, 0, 0, 0, 11, ...]
u_char * childrenArray(unsigned int *masks, unsigned int masks_size, u_char byte_number){
    unsigned int i, byte_mask, current_mask, offset;
    u_char *chldrn_arr = (u_char*)calloc(0, (BYTE_MAX+1)*sizeof(u_char));
    byte_mask = getByteMask(byte_number);
    offset = ((sizeof(int) - byte_number)*8);

    for(i=0; i<masks_size; i+=2){
        current_mask = masks[i] & byte_mask;
        current_mask = current_mask >> offset;

        chldrn_arr[current_mask] = 1;
    }

    return chldrn_arr;
}

// Create an array such that [key, bits, EndOfMask]  example:  [23, 0100100, 0, ...]
void byteArray(u_char *ba, unsigned int *masks,unsigned int masks_size, u_char byte_number){
    unsigned int i;
    u_char temp, offset;
    int j =2;
    for(i=0; i<masks_size; i++){
        if(i%2){
            if(masks[i] < 8){ //if prefix at this moment <8 means we mark it on an array as one of endings for this node
                offset = 7 - masks[i];
                temp = ba[j-1];
                u_char newval = (temp >> offset) | 1;
                newval = newval << offset;
                ba[j-1] = (ba[j-1] | newval);
            }
            else if(masks[i] == 8){
                ba[j] = 0;
                masks[i] -= 8;
            }
            else{
                ba[j] = 1;
                masks[i] -= 8;
            }
            j+=3;
        }
    }
}
// Count number of children to be created for current address Byte
u_char childrenCount(u_char *chldrnArr){
    int i;
    u_char counter =0;

    for(i=0; i<= BYTE_MAX; i++){
        if(chldrnArr[i])
            counter++;
    }
    return counter;
}
//Create a tree of masks from a root
void createTree(TreeNode *root, unsigned int *masks, unsigned int masks_size){
    unsigned int i;
    u_char byte_number = 1;
    u_char *chldrn_arr;
    u_char children_count = 0;
    printf("Check what children exist and count them!\n");
    chldrn_arr = childrenArray(masks, masks_size, byte_number); // get if what children with what keys exist
    children_count = childrenCount(chldrn_arr);
    //byte array
    u_char *ba= (u_char*)malloc((BYTE_MAX+1) * sizeof(u_char)* 3); // [0-255, 0000 0000, 0, 0-255, 0010 1000, 1]

    for(i=0; i<=BYTE_MAX*3; i+=3){ // 0-255
        if(chldrn_arr[i/3])
            ba[i] = i/3;
        else
            ba[i] = 0;
        ba[i+2] = 0;
    }
    printf("Byte magic!\n");
    byteArray(ba, masks, masks_size, byte_number);

    printf("Create the nodes of a tree\n");
    //based om ba create tree level CREATING A NODE AT CURRENT LEVEL
    if(root == nullptr){
        root = create_treenode(0,0,children_count);
    }else{
        root = create_treenode();
    }
    u_char *ba_stripped = (u_char*)malloc(children_count * 3 * sizeof(u_char)); //or counter + 1
    int j=0;
    for(i=0; i<BYTE_MAX*3; i+=3){
        if(ba[i] == 0 && ba[i+2] == 1){
            ba_stripped[j] = ba[i];
            ba_stripped[j+1] = ba[i+1];
            ba_stripped[j+2] = ba[i+2];
            j+=3;
        }
    }
    for(i=0; i< root->no_children; i++){
        root->children[i] = create_treenode(root);
    }
    root->no_children
    root->children[]
    printf("gg\n");
}





#endif
