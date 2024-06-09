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

std::vector<int> fn(std::vector<int>& arr, int k) {
  
  std::priority_queue<int, CRITERIA> heap;
  for (int num: arr) {
    heap.push(num);
    if (heap.size() > k) {
      heap.pop();
    }
  }
  std::vector<int> ans;
  while (heap.size() > 0) {
    ans.push_back(heap.top());
    heap.pop();
  }
  return ans;
}

int main() {

  return 0;
}
