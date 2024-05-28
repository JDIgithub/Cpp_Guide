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

/* 101. Symmetric Tree

Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

Example 1:

  Input: root = [1,2,2,3,4,4,3]
  Output: true

Example 2:

  Input: root = [1,2,2,null,3,null,3]
  Output: false
 
Constraints:

  The number of nodes in the tree is in the range [1, 1000].
  -100 <= Node.val <= 100
 
Follow up: Could you solve it both recursively and iteratively?
*/

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

// Recursive
bool compareMirrorTree(TreeNode* left, TreeNode * right){
  if(!left && !right) { 
    return true; 
  } else if(!left || !right) {
    return false;
  }
  if(left->val != right->val) return false;
  return compareMirrorTree(left->left, right->right) && compareMirrorTree(left->right, right->left);
}


bool isSymmetric(TreeNode* root) {
  return compareMirrorTree(root->left,root->right);
}

// Iterative with Queue
bool isSymmetricIterative(TreeNode* root) {
  if(!root) return true;
  std::queue<TreeNode*> que;
  TreeNode *left = root->left;
  TreeNode *right = root->right;

  que.push(root->left);
  que.push(root->right);
  while(!que.empty()){

    left=que.front();
    que.pop();
    right=que.front();
    que.pop();

    if(!left && !right)           continue;
    if(!left || !right)           return false;
    if(left->val!=right->val)     return false;

    que.push(left->left);     // left
    que.push(right->right);   // right
    que.push(left->right);    // left
    que.push(right->left);    // right
  }
  return true;
}



int main(){

  TreeNode *root = new TreeNode(1);
  root->left = new TreeNode(2);
  root->right = new TreeNode(2);
  root->right->left = new TreeNode(4);
  root->right->right = new TreeNode(3);
  root->left->left = new TreeNode(3);
  root->left->right = new TreeNode(4);
  
  auto xx = isSymmetric(root);
  std::cout << xx;
  return 0;
}