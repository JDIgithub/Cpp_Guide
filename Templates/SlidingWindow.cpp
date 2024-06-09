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

int fn(std::vector<int>& arr) {

  int left = 0, ans = 0, curr = 0;

  for (int right = 0; right < arr.size(); right++) {
    // Do logic here to add arr[right] to curr
    while (WINDOW_CONDITION_BROKEN) {
      // Remove arr[left] from curr
      left++;
    }
    // Update ans
  }
  return ans;
}




int main(){

  std::vector<int> nums {1,1,1,2,2,2,3,3};

  // auto xx = removeDuplicates(nums);

  // std::cout << xx;

  return 0;
}