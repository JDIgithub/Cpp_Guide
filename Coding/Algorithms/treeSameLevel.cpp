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

bool checkUtility(Node* node, int level, int *leafLevel) {
  if(node == nullptr){ return true;}

  // If a leaf node is encountered
  if(node->right == nullptr && node->left == nullptr){
    // Leaf node is found for the first time
    if(*leafLevel == 0){
      *leafLevel = level;
      return true;
    }
      // If it is not the first leaf, compare its level with the previous leaf's level
    return (level == *leafLevel);
  }
  return checkUtility(node->left,++level,leafLevel);  
}

bool check(Node* node){
  int level {0};
  int leafLevel {0};
  return checkUtility(node,level,&leafLevel);
}

int main() {

  Node *root = new Node(10);                     //               10        
  root->left = new Node(8);                     //          8        2 
  root->right = new Node(2);                    //       3     5   2
  root->left->left = new Node(3);
  root->left->right = new Node(5);
  root->right->left = new Node(2);

  if(check(root)){
    std::cout << "All leaves at the same level " << std::endl;
  } else {
    std::cout << "Some leaf has different level " << std::endl;
  }
}