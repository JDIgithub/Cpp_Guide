#include <string>
#include <algorithm>
#include <thread>
#include <iostream>
#include <functional>


// Binary Tree Node
struct Node {

  int data;
  Node *left,*right;
  Node(int value): data(value), left(nullptr), right(nullptr){}

};

// returns height of the tree
int height(Node* node) {
  if (node == nullptr) {
    return 0;
  } else {
    // Compute the height of each subtree
    int leftHeight = height(node->left);
    int rightHeight = height(node->right);
    // Use the larger one and add 1 for the current node
    return (leftHeight > rightHeight ? leftHeight : rightHeight) + 1;
  }
}

void printGivenLevel(Node * node, int level){

  if(node == nullptr) { return; }
  if(level == 1){ 
    std::cout << node->data << " "; 
  } else if (level > 1) {
    printGivenLevel(node->left, level - 1);
    printGivenLevel(node->right, level - 1);    
  }
}

// Print nodes of binary tree post order
// Left,Right,Root
void printLevelOrder (Node * node){

  int h = height(node);
  for(int i = 1; i <= h; ++i){
    printGivenLevel(node,i);
  }

}

int main() {

  Node *root = new Node(1);                     //               1        
  root->left = new Node(2);                     //          2        3 
  root->right = new Node(3);                    //       4     5
  root->left->left = new Node(4);
  root->left->right = new Node(5);

  std::cout << "Level-Order traversal: \n";
  printLevelOrder(root);                        // 1,2,3,4,5

}