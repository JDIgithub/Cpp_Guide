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

// LeetCode 26. 
/*

Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. 
The order of the elements may be changed. Then return the number of elements in nums which are not equal to val.
Consider the number of elements in nums which are not equal to val be k, to get accepted, you need to do the following things:
  Change the array nums such that the first k elements of nums contain the elements which are not equal to val. 
  The remaining elements of nums are not important as well as the size of nums.
  Return k.

Example 1:

Input: nums = [3,2,2,3], val = 3
Output: 2, nums = [2,2,_,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 2.
It does not matter what you leave beyond the returned k (hence they are underscores).

Example 2:

Input: nums = [0,1,2,2,3,0,4,2], val = 2
Output: 5, nums = [0,1,3,0,4,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
Note that the five elements can be returned in any order.
It does not matter what you leave beyond the returned k (hence they are underscores).

*/

// The best solution:
int removeElement(vector<int>& nums, int val) {
  int index = 0;
  for(int num: nums){
    if(num != val){       // If num is not to be deleted
      nums[index] = num;  // Put it into current index position and increment index
      index++;
    } // If num is to be deleted then just do not increment index and it will be either replaced or if its far enough in the array then it will be just left there
  }
  return index;
}


int main(){

  vector<int> nums {0,1,2,2,3,0,4,2};
  std::cout << removeElement(nums,2) << std::endl;

  for(int num: nums){
    std::cout << num << " ";
  }


  return 0;
}


