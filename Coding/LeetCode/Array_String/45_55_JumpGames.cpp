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

// 55. Jump Game
/*

You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.
Return true if you can reach the last index, or false otherwise.

Example 1:

Input: nums = [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.

*/

// Solution that kinda resembles Kadane's algorithm. 
// For every index, I'm checking the max reach I can have till that element, if that reach is less than the value of my index, 
// that means I can never reach this particular index and my answer should be false.
// tc : O(N)
// sc : O(1)

bool canJump(vector<int>& nums) {
        
  
  //it shows at max what index can I reach.
  //initially I can only reach index 0, hence reach = 0
  int reach = 0; 
  for(int idx = 0; idx < nums.size(); idx++) {
    //at every index I'll check if my reach was atleast able to 
    //reach that particular index.

    //reach >= idx -> great, carry on. Otherwise, 
    if(reach < idx) return false;
    
    //now as you can reach this index, it's time to update your reach
    //as at every index, you're getting a new jump length.
    reach = max(reach, idx + nums[idx]);
  }
  //this means that you reached till the end of the array, wohooo!! 
  return true;
}

// Or we can check maxIdx we can reach for every index like this:

bool canJump2(vector<int>& nums) {
  int maxIdx = nums[0];
  for (int i = 0; i < nums.size(); ++i) {
    if (maxIdx >= nums.size() - 1) return true;       // We can reach end
    if (nums[i] == 0 and maxIdx == i) return false;   // We got stuck on 0
    if (i + nums[i] > maxIdx) maxIdx = i + nums[i];   // If current value can give us bigger index reach 
  }
  return true;
}

// 45. Jump Game II
/*
You are given a 0-indexed array of integers nums of length n. You are initially positioned at nums[0].
Each element nums[i] represents the maximum length of a forward jump from index i. In other words, if you are at nums[i], you can jump to any nums[i + j] where:
0 <= j <= nums[i] and
i + j < n
Return the minimum number of jumps to reach nums[n - 1].
The test cases are generated such that you can reach nums[n - 1].

Example 1:

Input: nums = [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2. Jump 1 step from index 0 to 1, then 3 steps to the last index.
Example 2:

Input: nums = [2,3,0,1,4]
Output: 2


*/

int jump(vector<int>& nums) {

  for(int i = 1; i < nums.size(); i++) {
    nums[i] = max(nums[i] + i, nums[i-1]);    // Create array with jumpTo values
  }
  int jumpTo = 0;
  int count = 0;
  while(jumpTo < nums.size() - 1)
  {
    jumpTo = nums[jumpTo];      // Jump thourgh the jumpTo values and count jumps
    count++;
  }
  
  return count; 
}


int main(){

  std::vector<int> nums {1,1,1,1};
  std::cout << jump(nums);


  return 0;
}


