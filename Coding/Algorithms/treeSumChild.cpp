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


bool isSumProperty(Node* node) {

  int leftData,rightData {0};

  if(node == nullptr || (node->left == nullptr && node->right == nullptr) ) {
    return true;
  } else {

    if(node->left != nullptr){ leftData = node->left->data; }
    if(node->right != nullptr){ rightData = node->right->data; }

    if(node->data == (leftData + rightData) && isSumProperty(node->left) && isSumProperty(node->right)){
      return true;
    } else {
      return false;
    }
  }
}





int main() {

  Node *root = new Node(10);                     //               10        
  root->left = new Node(8);                     //          8        2 
  root->right = new Node(2);                    //       3     5   2
  root->left->left = new Node(3);
  root->left->right = new Node(5);
  root->right->left = new Node(2);

  if(isSumProperty(root)){
    std::cout << "Tree has this property" << std::endl;
  } else {
    std::cout << "Tree does not have this property" << std::endl;
  }

}