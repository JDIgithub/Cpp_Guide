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


// LeetCode 189. Rotate Array
/*

Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.

Example 1:

Input: nums = [1,2,3,4,5,6,7], k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]

*/

// My solution  Time: O(n)  Space: O(n)
void myRotate(vector<int>& nums, int k) {
  vector<int> storage(nums.size(),-1);

  for(int i = 0; i < nums.size(); i++){
    storage[(i+k)%nums.size()] = nums[i];
  }
  nums = storage;
}

// Revesing STL  Time: O(n)  Space: O(1)
void rotate(vector<int>& nums, int k) {

  std::reverse(nums.begin(),nums.end());
  std::reverse(nums.begin(),nums.begin()+k%nums.size());
  std::reverse(nums.begin()+k%nums.size(),nums.end());

}

// ToDo Reverse without STL function ?


int main(){

  vector<int> nums {1,2};
  rotate(nums,3);

  for(int num: nums){
    std::cout << num << " ";
  }


  return 0;
}


