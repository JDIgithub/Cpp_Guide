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

// 136. Single Number

/*
Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
You must implement a solution with a linear runtime complexity and use only constant extra space.

Example 1:

Input: nums = [2,2,1]
Output: 1
Example 2:

Input: nums = [4,1,2,1,2]
Output: 4
Example 3:

Input: nums = [1]
Output: 1
 
Constraints:

1 <= nums.length <= 3 * 104
-3 * 104 <= nums[i] <= 3 * 104
Each element in the array appears twice except for one element which appears only once.
*/

// XOR of two same numbers is 0
// If we will XOR through the array, the same numbers will cancel each out
// So only the single number will remains in the end

int singleNumber(std::vector<int>& nums) {

  if(nums.empty()) return 0;
  int single = nums[0];
  for(int i = 1; i < nums.size(); i++){
    single ^= nums[i];
  }

  return single;
}


int main(){

  std::vector<int> nums {4,1,2,1,2};
  std::cout << singleNumber(nums) << std::endl;  

  return 0;
}


