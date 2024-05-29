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

/* 112. Path Sum

Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path 
such that adding up all the values along the path equals targetSum.

 
Example 1:

  Input: root = [5,4,8,11,null,13,4,7,2,null,null,null,1], targetSum = 22
  Output: true
  Explanation: The root-to-leaf path with the target sum is shown.

Example 2:

  Input: root = [1,2,3], targetSum = 5
  Output: false
  Explanation: There two root-to-leaf paths in the tree:
  (1 --> 2): The sum is 3.
  (1 --> 3): The sum is 4.
  There is no root-to-leaf path with sum = 5.

Example 3:

  Input: root = [], targetSum = 0
  Output: false
  Explanation: Since the tree is empty, there are no root-to-leaf paths.
 

Constraints:

The number of nodes in the tree is in the range [0, 5000].
-1000 <= Node.val <= 1000
-1000 <= targetSum <= 1000

*/

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


bool hasPathSum(TreeNode* root, int targetSum) {

  if(!root) return false;
  // Check if this is a leaf
  if (!root->left && !root->right) {
    return targetSum == root->val;  
  }
  targetSum -= root->val;

  return hasPathSum(root->left,targetSum) || hasPathSum(root->right,targetSum);
}

int main(){

  TreeNode *root = new TreeNode(5);
  root->left = new TreeNode(4);
  root->right = new TreeNode(8);
  root->left->left = new TreeNode(11);
  root->left->left->left = new TreeNode(7);
  root->left->left->right = new TreeNode(2);
  root->right->left = new TreeNode(13);
  root->right->right = new TreeNode(4);
  root->right->right->right = new TreeNode(1);
  
  auto xx = hasPathSum(root,22);
  std::cout << xx;
  return 0;
}