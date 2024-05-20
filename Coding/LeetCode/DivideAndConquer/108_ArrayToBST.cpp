#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
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
#include <list>
#include <atomic>

using namespace std;

// 108. Convert sorted array to binary search tree
/*

Given an integer array nums where the elements are sorted in ascending order, convert it to a 
height-balanced binary search tree.

 
Example 1:


  Input: nums = [-10,-3,0,5,9]
  Output: [0,-3,9,-10,null,5]
  Explanation: [0,-10,5,null,-3,null,9] is also accepted:

Example 2:

  Input: nums = [1,3]
  Output: [3,1]
  Explanation: [1,null,3] and [3,1] are both height-balanced BSTs.
 

Constraints:

  1 <= nums.length <= 104
  -104 <= nums[i] <= 104
  nums is sorted in a strictly increasing order.
  
*/


struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};



TreeNode* createTree(vector<int> &nums, int left, int right) {
  if(left > right) return nullptr;

  int i = (left + right) / 2;
  TreeNode* root = new TreeNode(nums[i]);

  root->left = createTree(nums, left, i - 1);
  root->right = createTree(nums, i + 1, right);

  return root;
}

TreeNode* sortedArrayToBST(vector<int>& nums) {

    int left = 0;
    int right = nums.size() - 1;
    return createTree(nums, left, right);
}


int main(){

  std::vector<int> nums = {-10,-3,0,5,9};

  TreeNode* root = sortedArrayToBST(nums);


  return 0;
}






