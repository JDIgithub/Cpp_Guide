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

// 162. Find Peak Element

/*

A peak element is an element that is strictly greater than its neighbors.
Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.
You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.
You must write an algorithm that runs in O(log n) time.

 

Example 1:

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
Example 2:

Input: nums = [1,2,1,3,5,6,4]
Output: 5
Explanation: Your function can return either index number 1 where the peak element is 2, or index number 5 where the peak element is 6.
 

Constraints:

1 <= nums.length <= 1000
-231 <= nums[i] <= 231 - 1
nums[i] != nums[i + 1] for all valid i.
  
*/

// !! Thinking !!
// How to apply binary search here when the array is not sorted at the first glance?
// Well it is partially sorted..
// If there is a peak, it means that there is increasing part of the "hill" to the left of the peak
// And decreasing part of the "hill" to the right of the peak
//
//        Peak
//      ↑      ↓
//    ↑          ↓
//  ↑              ↓
//
// That means that we need to find out if we are in the increasing or decreasing part or if we hit the peak
// and move to the left or to the right accordingly




int findPeakElement(vector<int>& nums) {
        
  int i = 0;
  int left = 0;
  int right = nums.size()-1;


  while(left < right){

    i = (left + right)/2;

    if(nums[i] > nums[i + 1]){
      // We are in the decreasing part of the hill or peak
      // Peak must be to the left of nums[i + 1]
      if( (i == 0) || (nums[i] > nums[i - 1])){
        // We have a peak
        return i;
      } else {
        right = i - 1;
      }

    } else {
      // We are in the increasing part
      left = i + 1;
    }

  }

  return left;

}


int main(){

  std::vector<int> nums = {1,3,5,6,8};

  std::cout << findPeakElement(nums);

  return 0;
}






