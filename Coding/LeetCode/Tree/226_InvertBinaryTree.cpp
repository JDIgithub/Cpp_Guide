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
#include <future>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <math.h>
#include <stack>


using namespace std;

// 226. Invert Binary Tree

/*

Given the root of a binary tree, invert the tree, and return its root.

Example 1:

Input:

  root = [4,2,7,1,3,6,9]
        4
       / \
      2   7
     /|   |\
    1 3   6 9
    
Output: 

  newRoot [4,7,2,9,6,3,1]
        4
       / \
      7   2
     /|   |\
    9 6   3 1


Example 2:

Input: root = [2,1,3]
Output: [2,3,1]
Example 3:

Input: root = []
Output: []
 

Constraints:

The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100
*/



struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};



// Simple recursion with swapping childs
TreeNode* invertTree(TreeNode* root) {
        
  if(!root) return root;
  invertTree(root->left);
  invertTree(root->right);

  TreeNode *temp = root->left;
  root->left = root->right;
  root->right = temp;

  return root;        
}

int main(){

  TreeNode *root = new TreeNode(4);
  root->left = new TreeNode(2);
  root->right = new TreeNode(7);
  root->left->left = new TreeNode(1);
  root->left->right = new TreeNode(3);
  root->right->left = new TreeNode(6);
  root->right->right = new TreeNode(9);

  invertTree(root);

  return 0;
}


