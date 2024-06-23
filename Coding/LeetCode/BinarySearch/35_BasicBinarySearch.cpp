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

// 35. Search Insert Position

/*

Given a sorted array of distinct integers and a target value, return the index if the target is found. 
If not, return the index where it would be if it were inserted in order.
You must write an algorithm with O(log n) runtime complexity.

 

Example 1:

Input: nums = [1,3,5,6], target = 5
Output: 2

Example 2:

Input: nums = [1,3,5,6], target = 2
Output: 1

Example 3:

Input: nums = [1,3,5,6], target = 7
Output: 4
 

Constraints:

1 <= nums.length <= 104
-104 <= nums[i] <= 104
nums contains distinct values sorted in ascending order.
-104 <= target <= 104
  
*/


int searchInsert(vector<int>& nums, int target) {

  int mid = 0;
  int left = 0;
  int right = nums.size() - 1;

  if(target>nums[right]) return ++right;

  while(left <= right){

    mid = left + (right - left)/2;
    if(target < nums[mid]){
      right = mid - 1;
    } else if(target > nums[mid]) {
      left = mid + 1;
    } else {
      return mid;
    }
  }
  return left;
}


int main(){

  std::vector<int> nums = {1,3,5,6,8};

  std::cout << searchInsert(nums,2);

  return 0;
}






