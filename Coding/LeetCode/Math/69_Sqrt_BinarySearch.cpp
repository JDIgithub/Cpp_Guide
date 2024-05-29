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

/* 69. Sqrt(x)

Given a non-negative integer x, return the square root of x rounded down to the nearest integer. The returned integer should be non-negative as well.
You must not use any built-in exponent function or operator.
For example, do not use pow(x, 0.5) in c++ or x ** 0.5 in python.
 
Example 1:

  Input: x = 4
  Output: 2
  Explanation: The square root of 4 is 2, so we return 2.

Example 2:

  Input: x = 8
  Output: 2
  Explanation: The square root of 8 is 2.82842..., and since we round it down to the nearest integer, 2 is returned.
 

Constraints:

0 <= x <= 231 - 1

*/


// Using Binary search on range [0,X] but i do not need array because the indexes matches the value we need

int mySqrt(int x) {

  long long left = 0;
  long long right = x;
  long long mid;
  long long midPow;
  long long bestOption = 0;

  while(left <= right){
    mid = left + (right - left)/ 2;  
    midPow =  mid * mid;
    if(midPow == x) return mid;
    if(midPow > x){
      right = mid - 1;
    } else {
      bestOption = mid;
      left = mid + 1;
    }
  }
  return bestOption;
}


int main(){

  std::vector<int> nums {9,9,9,9};
  auto xx = mySqrt(2147395599);
  std::cout << xx;
  return 0;
}