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

using namespace std;

// 219. Contains Duplicate II
/*

Given an integer array nums and an integer k, return true if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

 

Example 1:

Input: nums = [1,2,3,1], k = 3
Output: true
Example 2:

Input: nums = [1,0,1,1], k = 1
Output: true
Example 3:

Input: nums = [1,2,3,1,2,3], k = 2
Output: false
*/

bool containsNearbyDuplicate(vector<int>& nums, int k) {

  std::unordered_map<int,int> duplicateMap;

  for(int i = 0; i < nums.size(); i++){ 
    if(duplicateMap.find(nums[i]) != duplicateMap.end()){
      if(std::abs(i - duplicateMap[nums[i]]) <= k){
        return true;
      } 
    }
    duplicateMap[nums[i]] = i;
  }
  return false;
}



int main(){

  std::vector<int> nums{1,2,3,1};
  auto result = containsNearbyDuplicate(nums,3);
  int pause = 0;
  
  return 0;
}






