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

// 100. Same Tree
/*

Given the roots of two binary trees p and q, write a function to check if they are the same or not.
Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

*/

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


bool isSameTree(TreeNode* p, TreeNode* q) {

  if(p == nullptr || q ==nullptr){ return p==q; }

  if(p->val != q->val || !isSameTree(p->left,q->left) || !isSameTree(p->right,q->right)){
    return false;
  }
  return true;    
}

int main(){

  TreeNode *root = new TreeNode(3);
  root->left = new TreeNode(9);
  root->right = new TreeNode(20);
  root->right->left = new TreeNode(15);
  root->right->right = new TreeNode(7);

  TreeNode *root2 = new TreeNode(3);
  root2->left = new TreeNode(9);
  root2->right = new TreeNode(20);
  root2->right->left = new TreeNode(15);
  root2->right->right = new TreeNode(7);


  std::cout << isSameTree(root,root2);


  return 0;
}


