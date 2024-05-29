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

/* 637. Average of levels in Binary Tree

Given the root of a binary tree, return the average value of the nodes on each level in the form of an array. Answers within 10-5 of the actual answer will be accepted.
 
Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: [3.00000,14.50000,11.00000]
Explanation: The average value of nodes on level 0 is 3, on level 1 is 14.5, and on level 2 is 11.
Hence return [3, 14.5, 11].
Example 2:

Input: root = [3,9,20,15,7]
Output: [3.00000,14.50000,11.00000]
 
Constraints:

The number of nodes in the tree is in the range [1, 104].
-231 <= Node.val <= 231 - 1

*/

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};


// BFS should be done iterativaly
// Simple BFS:
void bfs(TreeNode* root) {
  
  if (!root) return;
    
  std::queue<TreeNode*> q;
  q.push(root);
    
  while (!q.empty()) {
    TreeNode* current = q.front();
    q.pop(); 
    std::cout << current->val << " ";
    if (current->left) { q.push(current->left); }
    if (current->right) { q.push(current->right); }
  }
}

std::vector<double> averageOfLevelsMySolution(TreeNode* root) {

  if(!root) return {};
  std::vector<double> averages;
  std::queue<std::pair<TreeNode*,int>> q;
  q.push(std::make_pair(root,0));
  int count = 1;
  while(!q.empty()){
    TreeNode* current = q.front().first;  
    int level = q.front().second;
    if (level >= averages.size()) {
      averages.push_back(current->val);
    } else {
      averages[level] += current->val;
      count++;
    }
    q.pop();
    if (current->left) { q.push(std::make_pair(current->left,level+1)); }
    if (current->right) { q.push(std::make_pair(current->right,level+1)); }  

    if(q.front().second >= averages.size()){
      averages[level] = averages[level]/count;
      count = 1;
    }
  }

  return averages;
}

std::vector<double> averageOfLevels(TreeNode* root) {
  std::vector<double> averages;
  double sum = 0, avg = 0;

  std::queue<TreeNode*> q;
  q.push(root);

  while(!q.empty()) {
    avg = 0; 
    sum = 0;
    int size = q.size(), s = q.size();            // Size of the current level
    while(size--) {                               // This while loop goes only through the curent level and will create the next level
      TreeNode* node = q.front();
      q.pop();
      sum += node->val;

      if(node->left) { q.push(node->left); }
      if(node->right) { q.push(node->right); }
    }

    avg = sum / s;
    averages.push_back(avg);
  }

  return averages;
}

int main(){

  TreeNode *root = new TreeNode(3);
  root->left = new TreeNode(9);
  root->right = new TreeNode(20);
  root->right->left = new TreeNode(15);
  root->right->right = new TreeNode(7);

  
  auto xx = averageOfLevels(root);
  //std::cout << xx;
  return 0;
}