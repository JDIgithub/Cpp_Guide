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

// LeetCode 1. Two sum

vector<int> twoSum(vector<int>& nums, int target) {
        
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

vector<int> myTwoSum(vector<int>& nums, int target) {
        
  vector<int> indices; 
  for(int i = 0; i < nums.size() - 1; i++){
    for(int j = i+1; j < nums.size(); j++){
      if(target == (nums[i] + nums[j])) {
        indices.push_back(i);
        indices.push_back(j);
        return indices;
      }
    }
  }

  return indices;
}



int main(){


  return 0;
}


