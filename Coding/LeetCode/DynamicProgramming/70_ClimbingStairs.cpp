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

/* 70. Climbing Stairs

You are climbing a staircase. It takes n steps to reach the top.
Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

 
Example 1:

  Input: n = 2
  Output: 2
  Explanation: There are two ways to climb to the top.
  1. 1 step + 1 step
  2. 2 steps

Example 2:

  Input: n = 3
  Output: 3
  Explanation: There are three ways to climb to the top.
  1. 1 step + 1 step + 1 step
  2. 1 step + 2 steps
  3. 2 steps + 1 step

Constraints:

1 <= n <= 45

*/

int climbStairs(int n) {

  if(n == 0) return 0;
  if(n == 1) return 1;
  if(n == 2) return 2;

  // Starting with step 3
  int twoBack = 1;
  int oneBack = 2;
  int result;
  for(int i = 3; i <= n; i++){
    result = twoBack + oneBack;
    twoBack = oneBack;
    oneBack = result;
  }
  return result;
}

int main(){

  std::vector<int> nums {9,9,9,9};
  auto xx = climbStairs(5);
  std::cout << xx;
  return 0;
}