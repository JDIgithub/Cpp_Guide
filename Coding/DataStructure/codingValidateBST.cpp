#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <stack>
#include <cassert>

#include <iostream>
#include <limits>


struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

class Solution {
public:
  bool isValidBST(TreeNode* root) {
    return validate(root, std::numeric_limits<long long>::min(), std::numeric_limits<long long>::max());
  }

private:
  bool validate(TreeNode* node, long long min, long long max) {
    if (!node) return true;  
    if (node->val <= min || node->val >= max) return false;    
    return validate(node->left, min, node->val) && validate(node->right, node->val, max);
  }
};

int main() {
  // Construct a simple BST
  TreeNode *root = new TreeNode(2);
  root->left = new TreeNode(1);
  root->right = new TreeNode(3);

  Solution solution;
  bool result = solution.isValidBST(root);
    
  std::cout << "Is valid BST: " << std::boolalpha << result << std::endl; // Expected output: true

  // Clean up the allocated memory
  delete root->left;
  delete root->right;
  delete root;
  return 0;
}