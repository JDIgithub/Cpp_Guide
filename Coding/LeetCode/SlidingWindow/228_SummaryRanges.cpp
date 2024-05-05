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
using namespace std;

// 228. Summary Ranges

/*

You are given a sorted unique integer array nums.
A range [a,b] is the set of all integers from a to b (inclusive).
Return the smallest sorted list of ranges that cover all the numbers in the array exactly. That is, each element of nums is covered by exactly one of the ranges, 
and there is no integer x such that x is in one of the ranges but not in nums.

Each range [a,b] in the list should be output as:

"a->b" if a != b
"a" if a == b
 
Example 1:

Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: The ranges are:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"
Example 2:

Input: nums = [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
Explanation: The ranges are:
[0,0] --> "0"
[2,4] --> "2->4"
[6,6] --> "6"
[8,9] --> "8->9"
 

Constraints:

0 <= nums.length <= 20
-231 <= nums[i] <= 231 - 1
All the values of nums are unique.
nums is sorted in ascending order.

*/

std::vector<std::string> summaryRanges(std::vector<int>& nums) {

  if(nums.empty()) return {};
  std::vector<std::string> intervals;
  std::string interval = std::to_string(nums[0]);

  int leftWindow = 0;
  int rightWindow = 0;

  for(int i = 1; i < nums.size(); i++){

    if((nums[i-1]+1) == nums[i]){
      rightWindow = i;
    } else {
      if(leftWindow == rightWindow){
        intervals.push_back(std::to_string(nums[rightWindow]));
      } else {
        intervals.push_back(std::to_string(nums[leftWindow]) + "->" + std::to_string(nums[rightWindow]));
      }
      leftWindow = i;
      rightWindow = i;
    }
  }

  if(leftWindow == rightWindow){
    intervals.push_back(std::to_string(nums[rightWindow]));
  } else {
    intervals.push_back(std::to_string(nums[leftWindow]) + "->" + std::to_string(nums[rightWindow]));
  }


  return intervals;
}



int main(){


  std::vector<int> nums {-1};
  std::vector<std::string> intervals = summaryRanges(nums);
  
  for(auto interval: intervals){
    std::cout << interval << ",";
  }

  return 0;
}


