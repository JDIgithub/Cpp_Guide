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

bool CONDITION;



// Two pointers: one input, opposite ends
int fn(std::vector<int>& arr) {
  int left = 0;
  int right = int(arr.size()) - 1;
  int ans = 0;

  while (left < right) {
    // Do some logic here with left and right
    if (CONDITION) {
      left++;
    } else {
      right--;
    }
  }

  return ans;
}

// Two pointers: two inputs, exhaust both
int fn(std::vector<int>& arr1, std::vector<int>& arr2) {
  
  int i = 0, j = 0, ans = 0;
  while (i < arr1.size() && j < arr2.size()) {
    // do some logic here
    if (CONDITION) {
      i++;
    } else {
      j++;
    }
  }
  while (i < arr1.size()) {
    // Do logic
    i++;
  }
  while (j < arr2.size()) {
    // Do logic
    j++;
  }

  return ans;
}





int main(){

  std::vector<int> nums {1,1,1,2,2,2,3,3};

  // auto xx = removeDuplicates(nums);

  // std::cout << xx;

  return 0;
}