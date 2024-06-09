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



// Build a prefix sum
std::vector<int> fn(std::vector<int>& arr) {
  
  std::vector<int> prefix(arr.size());
  prefix[0] = arr[0];

  for (int i = 1; i < arr.size(); i++) {
    prefix[i] = prefix[i - 1] + arr[i];
  }

  return prefix;
}



int main(){

  std::vector<int> nums {1,1,1,2,2,2,3,3};

  // auto xx = removeDuplicates(nums);

  // std::cout << xx;

  return 0;
}