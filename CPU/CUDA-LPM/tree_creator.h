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

// sets up a children array of size 256 of{0,1} such that an index corresponds to a value of a mask [0, 0, 0, 3, 4, 0, 6, 0, 0, 0, 11, ...]
void childrenArray(u_char* chldrn_arr, unsigned int *masks, unsigned int masks_size, u_char byte_number){
    unsigned int i, byte_mask, current_mask, offset;

    byte_mask = getByteMask(byte_number);
    offset = ((sizeof(int) - byte_number)*8);

    for(i=0; i<masks_size; i+=2){
        current_mask = masks[i] & byte_mask;
        current_mask = current_mask >> offset;

        chldrn_arr[current_mask] = 1;
    }

}

// Create an array such that [key, bits, EndOfMask]  example:  [23, 0100100, 0, ...]
void byteArray(u_char *ba, unsigned int *masks,unsigned int masks_size, u_char byte_number){
    unsigned int i, tempi, byte_mask;
    u_char temp, offset;
    int j =0;
    for(i=0; i<masks_size; i++){
        if(i%2){
            byte_mask = getByteMask(byte_number);
            offset = (sizeof(int) - byte_number)*8;
            j = ((masks[i-1] & byte_mask)>> offset) *3; //proper value for a current BA member (Taking the #byte_number byte of mask)
            tempi = masks[i];

            if(masks[i] < 8){ //if prefix at this moment <8 means we mark it on an array as one of endings for this node
                offset = masks[i];
                temp = ba[j+1];
                u_char newval = (temp >> offset) | 1;
                newval = newval << offset;
                ba[j+1] = (ba[j+1] | newval);
                ba[j+2] = 1;
                tempi = ba[j+1];
            }
            else if(masks[i] == 8){
                masks[i] -= 8;
            }
            else{
                masks[i] -= 8;
            }
            printf("at%d, %d %d %d\n",j, ba[j], ba[j+1], ba[j+2]);
        }
    }
}

// Count number of children to be created for current address Byte
u_char childrenCount(u_char *chldrnArr){
    int i;
    u_char counter =0;

    for(i=0; i<= BYTE_MAX; i++){
        if(chldrnArr[i] == 1)
            counter++;
    }
    return counter;
}

//Create a tree of masks from a root
void createTree(TreeNode *root, unsigned int *masks, unsigned int masks_size){
    unsigned int i;
    u_char byte_number = 1;
    u_char children_count = 0;
    u_char *chldrn_arr = (u_char*)malloc((BYTE_MAX+1)*sizeof(u_char));
    for(i=0; i< BYTE_MAX+1; i++){
        chldrn_arr[i] = 0;
    }

    childrenArray(chldrn_arr, masks, masks_size, byte_number); // get if what children with what keys exist
    children_count = childrenCount(chldrn_arr);
    printf("There exist %d children for this node!\n", children_count);
    for(i=0;i<256; i++){
        printf("%d ", chldrn_arr[i]);
    }
    //byte array
    u_char *ba= (u_char*)malloc((BYTE_MAX+1) * sizeof(u_char)* 3); // [0-255, 0000 0000, 0, 0-255, 0010 1000, 1]

    for(i=0; i<=BYTE_MAX*3; i+=3){ // 0-255
        if(chldrn_arr[i/3] == 1){
            ba[i] = i/3;  // assign keys
            printf("children: %d exists\n", ba[i]);
        }
        else
            ba[i] = 0;

        ba[i+1] = 0;    // assign bits = {0000 0000}
        ba[i+2] = 0;    // assign endOfMask = false
    }
    printf("Byte magic!\n");
    byteArray(ba, masks, masks_size, byte_number);

    printf("Create the nodes of a tree\n");
    //based om ba create tree level CREATING A NODE AT CURRENT LEVEL
    if(root == NULL){
        root = create_treenode(0,0,children_count);
    }else{
        // create a child
    }

    u_char *ba_stripped = (u_char*)malloc(children_count * 3 * sizeof(u_char)); //or counter + 1
    int j=0;
    for(i=0; i<BYTE_MAX*3; i+=3){
        if(!(ba[i] == 0 && ba[i+1] == 0 && ba[i+2] == 1)){ //check if the Byte address appeared at all
           // printf("node key: %d\n", ba[i]);
            ba_stripped[j] = ba[i];
            ba_stripped[j+1] = ba[i+1];
            ba_stripped[j+2] = ba[i+2];
            j+=3;
        }
    }

    printf("1st level calculated\n");

    unsigned int temp;
    for(i=0;i<masks_size;i+=2){
        temp = masks[i] << 8;
        printf("%d.%d /%d\n", masks[i] >> 24, temp >> 24, masks[i+1]);
    }

}




#endif
