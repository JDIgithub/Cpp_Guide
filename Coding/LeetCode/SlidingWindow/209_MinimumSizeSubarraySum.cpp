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

// 209. Minimum Size SubArray Sum

/*

Given an array of positive integers nums and a positive integer target, return the minimal length of a 
subarray whose sum is greater than or equal to target. If there is no such subarray, return 0 instead.

 
Example 1:

Input: target = 7, nums = [2,3,1,2,4,3]
  Output: 2
  Explanation: The subarray [4,3] has the minimal length under the problem constraint.

Example 2:

  Input: target = 4, nums = [1,4,4]
  Output: 1

Example 3:

  Input: target = 11, nums = [1,1,1,1,1,1,1,1]
  Output: 0
 

Constraints:

  1 <= target <= 109
  1 <= nums.length <= 105
  1 <= nums[i] <= 104
  
*/
// O(n)
int minSubArrayLenN(int target, std::vector<int>& nums) {
  if(nums.empty()) return 0;

  int minLength = INT_MAX; // Use INT_MAX for clarity
  int leftEdge = 0;
  int sum = 0;

  for(int rightEdge = 0; rightEdge < nums.size(); rightEdge++) {
    sum += nums[rightEdge];

    // If the sum is big enough update minLength and try to shrink the window from the left 
    // If the sum will still be big enough after the reduction from the left, update minLength again
    // When the sum is not big enough anymore we need to expand to the right again
    while(sum >= target) {  
      minLength = std::min(minLength, rightEdge - leftEdge + 1);
      sum -= nums[leftEdge];
      leftEdge++;
    }
  }

  return minLength == INT_MAX ? 0 : minLength;
}

// O(n * log n)
/*
Binary Search Approach
The windowfind function checks if there exists a subarray of a given size whose sum is greater than or equal to the target.
It uses the sliding window technique to iterate through the array and maintain a window of the specified size.
If the sum of the elements in the window is greater than or equal to the target, it returns true.
Otherwise, it returns false.
The minSubArrayLen function finds the minimum length of a subarray whose sum is greater than or equal to the target using binary search.

It initializes a range from 1 to the size of the input array.
It repeatedly divides the range in half and checks if a subarray of the mid-point length satisfies the condition using the windowfind function.
If a valid subarray is found, it updates the upper bound of the range to mid-1 and stores the mid-point length as the minimum length found so far.
If a valid subarray is not found, it updates the lower bound of the range to mid+1.
The search continues until the lower bound is no longer less than or equal to the upper bound.
Finally, it returns the minimum length of the subarray.
The code efficiently utilizes the sliding window technique and binary search to find the minimum length subarray satisfying the given condition.

*/
bool windowfind(int length, std::vector<int>&nums, int target) {
  int sum = 0;
  int beginWindow = 0;
  int endWindow = 0;

  while( endWindow < nums.size() ){
    sum+=nums[endWindow];
    if( endWindow - beginWindow + 1 == length){
      if(sum >= target) return true;
      sum-=nums[beginWindow];
      beginWindow++;
    }
    endWindow++;
  }
  
  return false;
}

// low and high here is not a index but length ... there for low = 1 as min length possible
// Binary search is applied on Length !!!  
// We can see different length as ordered array... length = {1,2,3,4,5,6}
// So we can try particular length at a time if the sum is there or not
// If true goes into the lower half with shorter length
// else try the higher half
int minSubArrayLen(int target, std::vector<int>& nums) {
  int low = 1, high = nums.size(), mn = 0;
  while (low <= high) {
    int mid = (low + high) / 2;
    if (windowfind(mid, nums, target)) {
      high = mid-1;
      mn = mid;
    } else low = mid + 1;
  }
  return mn;
}


int main(){

  std::vector<int> nums = {2,3,1,2,4,3};

  std::cout << minSubArrayLen(7,nums);

  return 0;
}




