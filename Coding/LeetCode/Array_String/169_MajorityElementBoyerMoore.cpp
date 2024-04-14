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
using namespace std;

// LeetCode 169. Majority Element
/*

Given an array nums of size n, return the majority element.
The majority element is the element that appears more than ⌊n / 2⌋ times. 
You may assume that the majority element always exists in the array.

Example 1:

Input: nums = [3,2,3]
Output: 3
Example 2:

Input: nums = [2,2,1,1,1,2,2]
Output: 2

*/

// My solution
int MyMajorityElement(vector<int>& nums) {

  std::unordered_map<int,int> uniqueMap;
  for(int i = 0; i < nums.size(); i++){
    uniqueMap[nums[i]]++;
    if(uniqueMap[nums[i]] > (nums.size()/2)){
      return nums[i];
    }
  }
  return -1;
}

// To Do Boyer-Moore Voting Algorithm
/*
A more efficient method to find the majority element
This algorithm has a linear time complexity O(n) and requires only constant space O(1). Here’s how it works:

Initialize two variables: 
  One for storing a candidate element and another for counting. 
  Start by assuming the first element of the array as the potential majority candidate with a count of one.

Iterate through the array:
  For each element:
    If the count is zero, set the current element as the new candidate and set count to one.
    If the current element equals the candidate, increment the count.
    If the current element is different from the candidate, decrement the count.

Second Pass (optional but recommended for verification): 
  After determining the candidate, make another pass through the array to ensure that the candidate is indeed the majority element. 
  This is crucial in cases where no majority element exists.


*/

int majorityElement(vector<int>& nums) {
 
  int count = 0;
  int candidate = 0;

  // First pass to find the candidate
  for (int num : nums) {
    if (count == 0) {
      candidate = num;
    }
    count += (num == candidate) ? 1 : -1;
  }

  // Second pass to confirm the candidate (optional)
  count = 0;
  for (int num : nums) {
    if (num == candidate) {
      count++;
    }
  }

  if (count > nums.size() / 2) {
    return candidate;
  }

  return -1;  // this line is only reached if no majority element exists

}



// To Do Frequency Counter Pattern 


int main(){

  vector<int> nums {2,2,1,1,1,2,2};
  std::cout << majorityElement(nums) << std::endl;

  for(int num: nums){
    std::cout << num << " ";
  }


  return 0;
}


