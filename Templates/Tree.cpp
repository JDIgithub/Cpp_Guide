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

bool WINDOW_CONDITION_BROKEN;
class TreeNode{

public: 
  TreeNode * left;
  TreeNode * right;

};

// DFS recursive

int dfs(TreeNode* root) {
  if (root == nullptr) {
    return 0;
  }
  int ans = 0;
  // do logic
  dfs(root->left);
  dfs(root->right);
  return ans;
}

// DFS Iterative

int dfs(TreeNode* root) {
  std::stack<TreeNode*> stack;
  stack.push(root);
  int ans = 0;

  while (!stack.empty()) {
    TreeNode* node = stack.top();
    stack.pop();
    // do logic
    if (node->left != nullptr) {
      stack.push(node->left);
    }
    if (node->right != nullptr) {
      stack.push(node->right);
    }
  }
  return ans;
}

// BFS

int fn(TreeNode* root) {
  std::queue<TreeNode*> queue;
  queue.push(root);
  int ans = 0;
  while (!queue.empty()) {
    int currentLength = queue.size();
    // do logic for current level
    for (int i = 0; i < currentLength; i++) {
      TreeNode* node = queue.front();
      queue.pop();
      // do logic
      if (node->left != nullptr) {
        queue.push(node->left);
      }
      if (node->right != nullptr) {
        queue.push(node->right);
      }
    }
  }
 return ans;
}


int main() {
  std::vector<int> arr = {1, 2, 3, 1,3,2,5,4,7,9,1,4,5,1,1};
  int k = 6;
  std::cout << "Number of subarrays with sum " << k << " is: " << fn(arr, k) << std::endl;
  return 0;
}
