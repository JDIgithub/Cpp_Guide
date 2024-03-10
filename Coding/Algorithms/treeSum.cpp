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

int addTreeNode(Node* node) {

  if(node == nullptr){ return 0;}
  return (node->data + addTreeNode(node->left) + addTreeNode(node->right));

}

int main() {

  Node *root = new Node(10);                     //               10        
  root->left = new Node(8);                     //          8        2 
  root->right = new Node(2);                    //       3     5   2
  root->left->left = new Node(3);
  root->left->right = new Node(5);
  root->right->left = new Node(2);

  int sum = addTreeNode(root);
  std::cout << "Sum of all elements is: " << sum << std::endl;

}