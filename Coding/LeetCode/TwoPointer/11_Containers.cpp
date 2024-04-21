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

// 11. Container with the most water

/*

You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).
Find two lines that together with the x-axis form a container, such that the container contains the most water.
Return the maximum amount of water a container can store.
Notice that you may not slant the container.


Example 1:


Input: height = [1,8,6,2,5,4,8,3,7]
Output: 49
Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.
Example 2:

Input: height = [1,1]
Output: 1

*/

int maxArea(vector<int>& height) {
        
  if(height.size() < 2) return 0;

  int maxArea{0};
  int left{0};
  int right = height.size()-1;

  while(left < right){

    int newArea = ((right - left) * std::min(height[left],height[right]));

    if( newArea > maxArea ) {
      maxArea = newArea;
    }

    if(height[left] < height[right]){ // Move pointer which is pointing to the lower height
      left++;
    } else {
      right--;
    }
  }
  return maxArea;
}


int main(){

  std::vector<int> nums {1,8,6,2,5,25,8,25,7};
  std::cout << maxArea(nums);



  return 0;
}


