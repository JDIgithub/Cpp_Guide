#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <unordered_map>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <math.h>
using namespace std;

// 104. Maximum Depth of Binary Tree
/*

Given the root of a binary tree, return its maximum depth.
A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: 3
Example 2:

Input: root = [1,null,2]
Output: 2


*/

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

// We need to use recursion for both left and right child
int maxDepth(TreeNode* root) {

  if(root== NULL){ return 0; }          // Node is empty so depth = 0
  int left = maxDepth(root->left);      // To get the Depth of the left sub-tree
  int right = maxDepth(root->right);    // To get the Depth of the right sub-tree
  return max(left,right)+1;             // Choose if depth is higher on the left or right sub-tree and increment the higher depth

}


int main(){

  TreeNode *root = new TreeNode(3);
  root->left = new TreeNode(9);
  root->right = new TreeNode(20);
  root->right->left = new TreeNode(15);
  root->right->right = new TreeNode(7);
  std::cout << maxDepth(root);


  return 0;
}


