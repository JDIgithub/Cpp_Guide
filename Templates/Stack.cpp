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

// Monotonic increasing stack
// The same logic can be applied to maintain a monotonic queue.

int fn(std::vector<int>& arr) {
  std::stack<int> stack;
  int ans = 0;

  for (int num: arr) {
    // for monotonic decreasing, just flip the > to <
    while (!stack.empty() && stack.top() > num) {
      // do logic
      stack.pop();
    }

     stack.push(num);
  }
}

int main() {
  std::vector<int> arr = {1, 2, 3, 1,3,2,5,4,7,9,1,4,5,1,1};
  int k = 6;
  std::cout << "Number of subarrays with sum " << k << " is: " << fn(arr, k) << std::endl;
  return 0;
}
