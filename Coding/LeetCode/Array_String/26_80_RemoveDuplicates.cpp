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

// LeetCode 26. Remove Duplicates from Sorted Array 
/*

Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. 
The relative order of the elements should be kept the same. Then return the number of unique elements in nums.
Consider the number of unique elements of nums to be k, to get accepted, you need to do the following things:
  Change the array nums such that the first k elements of nums contain the unique elements in the order they were present in nums initially. 
  The remaining elements of nums are not important as well as the size of nums.
  Return k.

Example 1:

Input: nums = [1,1,2]
Output: 2, nums = [1,2,_]
Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).
Example 2:

Input: nums = [0,0,1,1,1,2,2,3,3,4]
Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
It does not matter what you leave beyond the returned k (hence they are underscores).

*/

// The STL algorithm std::unique
// Runtime 7 ms Memory 21.16 MB
int removeDuplicatesSTL(vector<int>& nums) {
  return std::unique(nums.begin(),nums.end()) - nums.begin(); 
}

// Runtime 5ms Memory 21.02 MB
int removeDuplicates(vector<int>& nums) {
  int index=1;
  for(int i=1; i<nums.size(); i++){
    if(nums[i]!=nums[i-1]){ // We can do this because its ordered array so the duplicates must be next to each other
      nums[index]=nums[i];
      index++;
    }

  }
  return index;
}


int main(){

  vector<int> nums {0,1,2,2,3};
  std::cout << removeDuplicates(nums) << std::endl;

  for(int num: nums){
    std::cout << num << " ";
  }


  return 0;
}



/* 80. Remove Duplicates from Sorted Array II

Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. 
The relative order of the elements should be kept the same. Since it is impossible to change the length of the array in some languages,
you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates,
then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.
Return k after placing the final result in the first k slots of nums.
Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

The judge will test your solution with the following code:

int[] nums = [...]; // Input array
int[] expectedNums = [...]; // The expected answer with correct length

int k = removeDuplicates(nums); // Calls your implementation

assert k == expectedNums.length;
for (int i = 0; i < k; i++) {
    assert nums[i] == expectedNums[i];
}
If all assertions pass, then your solution will be accepted.

 

Example 1:

  Input: nums = [1,1,1,2,2,3]
  Output: 5, nums = [1,1,2,2,3,_]
  Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
  It does not matter what you leave beyond the returned k (hence they are underscores).

Example 2:

  Input: nums = [0,0,1,1,1,1,2,3,3]
  Output: 7, nums = [0,0,1,1,2,3,3,_,_]
  Explanation: Your function should return k = 7, with the first seven elements of nums being 0, 0, 1, 1, 2, 3 and 3 respectively.
  It does not matter what you leave beyond the returned k (hence they are underscores).
 

Constraints:

  1 <= nums.length <= 3 * 104
  -104 <= nums[i] <= 104
  nums is sorted in non-decreasing order.

*/


int removeDuplicates2(std::vector<int>& nums) {

  int i =0;
  
  for(int num : nums) {

    // i == 0: This condition ensures that the first element is always included in the modified vector. 
    // i == 1: This condition ensures that the second element is always included in the modified vector.
    // nums[i-2] != num: This condition checks if the current element is not the same as the element two positions before the current position i. 
    // This ensures that only two occurrences of any element are included in the modified vector.
    if(i==0 || i==1 || nums[i-2] != num) {
      nums[i] = num;
      i++;
    }
  }
  return i;
}



int main2(){

  std::vector<int> nums {1,1,1,2,2,2,3,3};

  auto xx = removeDuplicates2(nums);

  std::cout << xx;

  return 0;
}