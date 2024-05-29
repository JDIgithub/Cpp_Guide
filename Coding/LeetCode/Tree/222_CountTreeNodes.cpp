#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <mutex>
#include <shared_mutex>
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>
#include <list>
#include <random>
#include <atomic>


using namespace std::literals;

/* 222. Count Complete Tree Node

Given the root of a complete binary tree, return the number of the nodes in the tree.
According to Wikipedia, every level, except possibly the last, is completely filled in a complete binary tree
and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

Design an algorithm that runs in less than O(n) time complexity.

 

Example 1:


Input: root = [1,2,3,4,5,6]
Output: 6
Example 2:

Input: root = []
Output: 0
Example 3:

Input: root = [1]
Output: 1
 

Constraints:

The number of nodes in the tree is in the range [0, 5 * 104].
0 <= Node.val <= 5 * 104
The tree is guaranteed to be complete.

*/

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


/* One Solution

int height(TreeNode *root){
  if(!root) return 0;
  int h = 1;
  while(root->left){            
    root = root->left; 
    h++;
  }                            
  return h;
}

int countNodes(TreeNode* root, int current = 1) {
  if(!root) return 0;
  int lh = height(root->left);
  int rh = height(root->right);
    
  if (rh == 0) {
    if (root->left) {
      return current * 2;
    } else {
      return current;
    }
  } else {
    if (lh == rh) {
      return countNodes(root->right, current * 2 + 1);
    } else {
      return countNodes(root->left, current * 2);
    }
  }
}

*/

/*

A more efficient way to count nodes in a complete binary tree is to use binary search. The main idea is to:
  
  Compute the height of the tree.
  Use binary search to count the nodes on the last level.

*/


// Function to calculate the height of the tree
int height(TreeNode* root) {
  int h = 0;
  while (root) {
    root = root->left;
    h++;
  }
  return h;
}

// Function to check if a node exists at the given index
bool nodeExists(int index, int height, TreeNode* node) {
  int left = 0, right = pow(2, height) - 1;
  int mid;
  for (int i = 0; i < height; ++i) {
    mid = left + (right - left) / 2;
    if (index <= mid) {       // If desired index is to the left of the middle (<) we need to go to the left
      node = node->left;
      right = mid;            // If we are going to the left subtree the new right of that subtree is previous tree's middle
    } else {                  // Else we need to go to the right  
      node = node->right;
      left = mid + 1;         // If we are going to the right subtree the new left of that subtree is previous tree's (middle + 1)
    }
  }
  return node != nullptr;
}

int countNodes(TreeNode* root) {
  
  if (!root) return 0;
  int h = height(root) - 1; // Height of the full filled binary tree
  if (h < 0) return 0; // Empty tree

  int left = 0, right = pow(2, h) - 1;
  int last_level_count = 0;

  // Binary search to find the number of nodes at the last level
  // We can think of the leaves as elements of array and binary search on them if the middle element exist go to the right subtree
  // else go to the left subtree
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (nodeExists(mid, h, root)) {
      last_level_count = mid + 1;
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  // Total number of nodes is the sum of all nodes in the full levels and last level
  return pow(2, h) - 1 + last_level_count;
}



int main(){

  TreeNode *root = new TreeNode(1);
  root->left = new TreeNode(2);
  root->left->left = new TreeNode(4);
  root->left->right = new TreeNode(5);
  root->right = new TreeNode(3);
  root->right->left = new TreeNode(6);

  
  auto xx = countNodes(root);
  std::cout << xx;
  return 0;
}