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

int binarySearch(std::vector<int>& arr, int target) {
  int left = 0;
  int right = int(arr.size()) - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) {
      // do something;
      return mid;
    }
    if (arr[mid] > target) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }      
  // left is the insertion point
  return left;
}

// Binary search: duplicate elements, left-most insertion point
// The second function’s initialization with right = arr.size() and the condition left < right combined with the update logic ensures that 
// the left-most insertion point is found. This is useful in scenarios where we need to handle duplicates or ensure the smallest possible index for insertion,
// even at the end of the array.
int binarySearch2(std::vector<int>& arr, int target) {
  int left = 0;
  int right = arr.size();
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] >= target) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }  
  return left;
}

// Binary search: duplicate elements, right-most insertion point
int binarySearch3(std::vector<int>& arr, int target) {
  int left = 0;
  int right = arr.size();
  while (left < right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] > target) {
      right = mid;
    } else {
      left = mid + 1;
    }
  }  
  return left;
}

// Binary search: for greedy problems
// If looking for a minimum
int MINIMUM_POSSIBLE_ANSWER;
int MAXIMUM_POSSIBLE_ANSWER;
bool BOOLEAN;

int fn(std::vector<int>& arr) {
  int left = MINIMUM_POSSIBLE_ANSWER;
  int right = MAXIMUM_POSSIBLE_ANSWER;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (check(mid)) {
      right = mid - 1;
    } else {
      left = mid + 1;
    }
  }
  return left;
}

bool check(int x) {
  // this function is implemented depending on the problem
  return BOOLEAN;
}

// If looking for a maximum
int fn(std::vector<int>& arr) {
  int left = MINIMUM_POSSIBLE_ANSWER;
  int right = MAXIMUM_POSSIBLE_ANSWER;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (check(mid)) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return right;
}

bool check(int x) {
  // this function is implemented depending on the problem
  return BOOLEAN;
}




int main() {

  return 0;
}
