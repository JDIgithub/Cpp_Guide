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

// Print nodes of binary tree post order
// Left,Right,Root
void printPostOrder (Node * node){
  if(node == nullptr){
    return;
  }

  // first recur on left subtree
  printPostOrder(node->left);
  // then recur on right subtree
  printPostOrder(node->right);
  // then print data
  std::cout << node->data << '\n';
}

// Print nodes of binary tree pre order
// Root,Left,Right
void printPreOrder (Node * node){
  if(node == nullptr){
    return;
  }
  // first print data
  std::cout << node->data << '\n';
  // then recur on left subtree
  printPreOrder(node->left);
  // then recur on right subtree
  printPreOrder(node->right);
}

// Print nodes of binary tree in order
// Left,Root,Right
void printInOrder (Node * node){
  if(node == nullptr){
    return;
  }
  // first recur on left subtree
  printInOrder(node->left);
  // then print data
  std::cout << node->data << '\n';
  // then recur on right subtree
  printInOrder(node->right);
}

int main() {

  Node *root = new Node(1);                     //               1        
  root->left = new Node(2);                     //          2        3 
  root->right = new Node(3);                    //       4     5
  root->left->left = new Node(4);
  root->left->right = new Node(5);

  std::cout << "Pre-Order traversal: \n";
  printPreOrder(root);                        // 1,2,4,5,3
  std::cout << "In-Order traversal: \n";
  printInOrder(root);                         // 4,2,5,1,3
  std::cout << "Post-Order traversal: \n";
  printPostOrder(root);                       // 4,5,2,3,1

}