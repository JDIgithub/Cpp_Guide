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

bool WINDOW_CONDITION_BROKEN;
class ListNode{

public: 
  ListNode * next;

};



// Find number of sub-arrays that fit an exact criteria
// To Do: Use this in some exercise
int fnE(std::vector<int>& arr, int k) {

  std::unordered_map<int, int> counts;
  counts[0] = 1;
  int ans = 0, curr = 0;
  for (int num: arr) {
    // Do logic to change curr
    ans += counts[curr - k];
    counts[curr]++;
  }

  return ans;
}

// Example
/*
Exercise: Number of Sub-arrays with Sum Equals K
Given an array of integers arr and an integer k, write a function to find the number of continuous sub-arrays that sum up to k.
*/


int fn(std::vector<int>& arr, int k) {
  std::unordered_map<int, int> counts;
  counts[0] = 1;
  int ans = 0, curr = 0;
  for (int num: arr) {
    curr += num; // Update current prefix sum
    ans += counts[curr - k]; // Check if there is a prefix sum that matches curr - k
    counts[curr]++; // Update the counts of the current prefix sum
  }
  return ans;
}

int main() {
  std::vector<int> arr = {1, 2, 3, 1,3,2,5,4,7,9,1,4,5,1,1};
  int k = 6;
  std::cout << "Number of subarrays with sum " << k << " is: " << fn(arr, k) << std::endl;
  return 0;
}
