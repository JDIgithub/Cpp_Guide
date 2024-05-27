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

// 1. Two Sum
/*

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.

*/

std::vector<int> twoSum(vector<int>& nums, int target) {
    
  // with HashMap
  unordered_map<int, int> mp;
  for(int i = 0; i < nums.size(); i++){
    if(mp.find(target - nums[i]) == mp.end()){
      // If not, add the current number and its index to the map
      mp[nums[i]] = i;
    } else {
      // If yes, return the indices of the current number and its complement
      return {mp[target - nums[i]], i};
    }
  }
  return {-1,-1};
}


int main(){

  std::vector<int> nums{2,7,11,15};
  auto result = twoSum(nums, 9);

  int pause = 0;
  
  return 0;
}






