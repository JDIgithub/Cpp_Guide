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

/* 198. House Robber

You are a professional robber planning to rob houses along a street. 
Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected
and it will automatically contact the police if two adjacent houses were broken into on the same night.
Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.


Example 1:

Input: nums = [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
Total amount you can rob = 1 + 3 = 4.
Example 2:

Input: nums = [2,7,9,3,1]
Output: 12
Explanation: Rob house 1 (money = 2), rob house 3 (money = 9) and rob house 5 (money = 1).
Total amount you can rob = 2 + 9 + 1 = 12.
 

Constraints:

1 <= nums.length <= 100
0 <= nums[i] <= 400

*/


// ToDo Some decent explaining in comments please
int rob(std::vector<int>& nums) {
  if (nums.empty()) return 0;
  int prev1 = 0;
  int prev2 = 0;
  for (int num : nums) {
    int tmp = prev1;
    prev1 = std::max(prev2 + num, prev1);
    prev2 = tmp;
  }
  return prev1;
}



int main(){

  std::vector<int> nums {4,9,7,1,7};
  auto xx = rob(nums);
  std::cout << xx;
  return 0;
}