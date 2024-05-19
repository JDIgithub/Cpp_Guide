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

// 530. Minimum Absolute Difference in BST

/*

Given the root of a Binary Search Tree (BST), return the minimum absolute difference between the values of any two different nodes in the tree.


Example 1:

Input: root = [4,2,6,1,3]
Output: 1
Example 2:


Input: root = [1,0,48,null,null,12,49]
Output: 1
 

Constraints:

The number of nodes in the tree is in the range [2, 104].
0 <= Node.val <= 105
 

*/

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


void dfs(TreeNode* root, int& result) {
  
  static TreeNode* prev{nullptr}; // prev is local to the function but retains its value across recursive calls.

  if (!root) return;

  dfs(root->left, result);
  if (prev) { result = std::min(result, std::abs(prev->val - root->val)); }
  prev = root;
  dfs(root->right, result);

}

int getMinimumDifference(TreeNode* root) {
  int result = std::numeric_limits<int>::max();
  dfs(root, result);
  return result;
}

// Maybe better solution is to pass the previous node as a parameter -> Thread safety, Clarity, etc..
/*
void dfs(TreeNode* root, TreeNode*& prev, int& result) {
  if (!root) return;

  dfs(root->left, prev, result);
  if (prev) { result = std::min(result, std::abs(prev->val - root->val)); }
  prev = root;
  dfs(root->right, prev, result);
}

int getMinimumDifference(TreeNode* root) {
  int result = std::numeric_limits<int>::max();
  TreeNode* prev = nullptr;
  dfs(root, prev, result);
  return result;
}

*/


int main(){

  TreeNode *root = new TreeNode(4);
  root->left = new TreeNode(2);
  root->right = new TreeNode(6);
  root->left->left = new TreeNode(1);
  root->left->right = new TreeNode(3);
  //root->right->left = new TreeNode(6);
  //root->right->right = new TreeNode(9);


  std::cout << getMinimumDifference(root);

  return 0;
}






