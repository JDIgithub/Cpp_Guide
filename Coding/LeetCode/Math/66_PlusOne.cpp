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


using namespace std::literals;

/* 66. Plus One

You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. 
The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.
Increment the large integer by one and return the resulting array of digits.

 
Example 1:

  Input: digits = [1,2,3]
  Output: [1,2,4]
  Explanation: The array represents the integer 123.
  Incrementing by one gives 123 + 1 = 124.
  Thus, the result should be [1,2,4].

Example 2:

  Input: digits = [4,3,2,1]
  Output: [4,3,2,2]
  Explanation: The array represents the integer 4321.
  Incrementing by one gives 4321 + 1 = 4322.
  Thus, the result should be [4,3,2,2].

Example 3:

  Input: digits = [9]
  Output: [1,0]
  Explanation: The array represents the integer 9.
  Incrementing by one gives 9 + 1 = 10.
  Thus, the result should be [1,0].
 

Constraints:

1 <= digits.length <= 100
0 <= digits[i] <= 9
digits does not contain any leading 0's.

*/

std::vector<int> plusOneShort(std::vector<int>& digits) {
     
  if(digits.empty()) return {};   
  std::vector<int> result;
  int number = 0;
  int digitMagnitude = digits.size() - 1;
  for(int digit: digits){
    number += (digit * std::pow(10,digitMagnitude));
    digitMagnitude--;
  }

  number++;
  digitMagnitude = std::log10(number);

  while(digitMagnitude >= 0){
    int magnitude = std::pow(10,digitMagnitude);
    int toPush = number / magnitude;
    result.push_back(toPush);
    number = number % magnitude;
    digitMagnitude--;

  }

  return result;   
}


std::vector<int> plusOne(std::vector<int>& digits) {
     
  if(digits.empty()) return {};   
  
  for(int i = digits.size() - 1; i >= 0; i--){
   
    if(digits[i] < 9){
      digits[i]++;
      return digits;
    } else {
      digits[i] = 0; // As it was 9
    }
  }
  // If we got here it means that all the digits were 9 so now they are all 0 and we need to add 1
  std::vector<int> result;
  result.push_back(1);
  for(auto digit: digits){
    result.push_back(digit);  // Adding 0's
  }
  return result;  
}


int main(){

  std::vector<int> nums {9,9,9,9};
  auto xx = plusOne(nums);
  //std::cout << xx;
  return 0;
}